import websocket
import json
import threading
import time
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from binance.client import Client
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import os
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Technical Indicator Calculations
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def compute_bollinger_width(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return (upper_band - lower_band) / sma

def compute_rolling_volatility(series, window=20):
    return series.pct_change().rolling(window=window).std()

# Trading Action Logger
class TradingActionLogger:
    def __init__(self, log_file='trading_actions.log'):
        self.log_file = log_file
        self.action_history = []
        with open(self.log_file, 'w') as f:
            f.write("Timestamp,Action,Description,LogReturn,OrderFlowImbalance,ScaledVolatility,MACD,BollingerWidth,RollingVolatility,Reason\n")

    def log_action(self, action, features, timestamp=None):
        # Use provided timestamp or fallback to current time
        timestamp = timestamp if timestamp else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        action_desc = {0: 'hold', 1: 'buy', 2: 'sell'}[action]
        reason = self.get_reason(action, features)
        log_entry = f"{timestamp},{action},{action_desc},{features.get('log_return', 0):.4f},{features.get('order_flow_imbalance', 0):.4f},{features.get('scaled_volatility', 0):.4f},{features.get('macd', 0):.4f},{features.get('bollinger_width', 0):.4f},{features.get('rolling_volatility', 0):.4f},{reason}"
        self.action_history.append(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(f"{log_entry}\n")
        logger.info("Trading Action History:")
        for entry in self.action_history[-5:]:
            logger.info(f"  {entry}")

    def get_reason(self, action, features):
        lr = features.get('log_return', 0)
        ofi = features.get('order_flow_imbalance', 0)
        macd = features.get('macd', 0)
        bw = features.get('bollinger_width', 0)
        if action == 1 and (lr > 0 or ofi > 0 or macd > 0):
            return f"Positive return ({lr:.4f}), bullish flow ({ofi:.4f}), or MACD crossover ({macd:.4f})"
        elif action == 2 and (lr < 0 or bw > 0.5 or macd < 0):
            return f"Negative return ({lr:.4f}), high volatility, or wide Bollinger ({bw:.4f})"
        return "No clear trend"

# Binance WebSocket Client
class BinanceWebSocketClient:
    def __init__(self, symbol, processor):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        self.ws = None
        self.running = False
        self.processor = processor

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            trade = {
                'timestamp': datetime.fromtimestamp(data['T'] / 1000).isoformat(),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data['m']
            }
            self.processor.on_trade(trade)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.running:
            logger.info("Attempting to reconnect...")
            self.reconnect()

    def on_open(self, ws):
        logger.info("WebSocket connection opened")

    def connect(self):
        self.running = True
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def reconnect(self):
        time.sleep(2)
        self.connect()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

# Data Preprocessing
class RealTimeProcessor:
    def __init__(self, symbol, ADV_minute, ADVOL_minute, scaler, callback=None):
        self.symbol = symbol
        self.ADV_minute = ADV_minute
        self.ADVOL_minute = ADVOL_minute
        self.scaler = scaler
        self.trade_buffer = defaultdict(list)
        self.last_processed_minute = None
        self.callback = callback
        self.feature_cols = ['log_return', 'scaled_volume', 'order_flow_imbalance', 'scaled_volatility', 'num_trades', 'hour', 'day', 'rsi', 'macd', 'bollinger_width', 'rolling_volatility']
        self.price_buffer = []
        self.minute_prices = []
        self.last_features = None

    def on_trade(self, trade):
        trade_time = pd.to_datetime(trade['timestamp'])
        minute = trade_time.floor('min')
        self.trade_buffer[minute].append(trade)
        self.price_buffer.append(float(trade['price']))
        if len(self.price_buffer) > 26:
            self.price_buffer.pop(0)

        if self.last_processed_minute is not None and minute > self.last_processed_minute:
            features = self.process_minute(self.last_processed_minute)
            if features:
                self.minute_prices.append(features['close_price'])
                if len(self.minute_prices) > 26:
                    self.minute_prices.pop(0)
                self.last_features = features
                if self.callback:
                    self.callback(features)
            self.last_processed_minute = minute
        elif self.last_processed_minute is None:
            self.last_processed_minute = minute

    def process_minute(self, minute):
        trades = self.trade_buffer.get(minute, [])
        if not trades:
            logger.warning(f"No trades for minute {minute}")
            return None

        prices = [float(t['price']) for t in trades]
        quantities = [float(t['quantity']) for t in trades]
        is_buyer_maker = [t['is_buyer_maker'] for t in trades]

        open_price = prices[0]
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]

        buy_volume = sum(q for q, m in zip(quantities, is_buyer_maker) if not m)
        sell_volume = sum(q for q, m in zip(quantities, is_buyer_maker) if m)
        total_volume = buy_volume + sell_volume

        order_flow_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        num_trades = np.log1p(len(trades))
        volatility = np.std(prices) if len(prices) > 1 else 0
        log_return = np.log(close_price / open_price) if open_price > 0 and close_price > 0 else 0
        scaled_volume = total_volume / self.ADV_minute if self.ADV_minute > 0 else 0
        scaled_volatility = volatility / self.ADVOL_minute if self.ADVOL_minute > 0 else 0
        hour = minute.hour / 23.0
        day = minute.weekday() / 6.0

        rsi = 50.0
        macd = 0.0
        bollinger_width = 0.0
        rolling_volatility = 0.0
        if len(self.price_buffer) >= 14:
            price_series = pd.Series(self.price_buffer)
            rsi = compute_rsi(price_series).iloc[-1]
            if np.isnan(rsi):
                rsi = 50.0
        if len(self.minute_prices) >= 26:
            minute_series = pd.Series(self.minute_prices)
            macd = compute_macd(minute_series).iloc[-1]
            bollinger_width = compute_bollinger_width(minute_series).iloc[-1]
            rolling_volatility = compute_rolling_volatility(minute_series).iloc[-1]
            if np.isnan(macd):
                macd = 0.0
            if np.isnan(bollinger_width):
                bollinger_width = 0.0
            if np.isnan(rolling_volatility):
                rolling_volatility = 0.0

        features = {
            'log_return': log_return,
            'scaled_volume': scaled_volume,
            'order_flow_imbalance': order_flow_imbalance,
            'scaled_volatility': scaled_volatility,
            'num_trades': num_trades,
            'hour': hour,
            'day': day,
            'rsi': rsi / 100.0,
            'macd': macd,
            'bollinger_width': bollinger_width,
            'rolling_volatility': rolling_volatility,
            'close_price': close_price,
            'timestamp': minute  # Store timestamp for logging
        }

        feature_vector = pd.DataFrame([features], columns=self.feature_cols)
        normalized_features = self.scaler.transform(feature_vector)[0]
        normalized_features_dict = {key: val for key, val in zip(self.feature_cols, normalized_features)}
        normalized_features_dict['close_price'] = close_price
        normalized_features_dict['timestamp'] = minute

        # Validate features
        if np.any(np.isnan(list(normalized_features_dict.values())[:-2])):  # Exclude close_price, timestamp
            logger.warning(f"NaN detected in features for {minute}: {normalized_features_dict}")
        logger.info(f"Processed features for {minute}: {normalized_features_dict}")
        return normalized_features_dict

# JEPA Model and MPC
def vicreg_loss(z1, z2, gamma=1.0, lambda_=1.0, mu=1.0):
    inv_loss = nn.MSELoss()(z1, z2)
    if z1.shape[0] < 2:
        return lambda_ * inv_loss
    epsilon = 1e-6
    var_z1 = torch.var(z1, dim=0, correction=1) + epsilon
    var_z2 = torch.var(z2, dim=0, correction=1) + epsilon
    var_loss = torch.mean(torch.relu(1 - var_z1)) + torch.mean(torch.relu(1 - var_z2))
    cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov_z2 = (z2.T @ z2) / (z1.shape[0] - 1)
    cov_loss = (cov_z1.pow(2).sum() - cov_z1.diagonal().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov_z2.pow(2).sum() - cov_z2.diagonal().pow(2).sum()) / z1.shape[1]
    return lambda_ * inv_loss + gamma * var_loss + mu * cov_loss

class JEPAModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, pred_steps):
        super(JEPAModel, self).__init__()
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.2, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, input_dim * pred_steps)
        )
        self.reg_proj = nn.Linear(d_model, input_dim)
        self.pred_steps = pred_steps
        self.d_model = d_model

    def forward(self, x):
        x = self.input_proj(x)
        z_past = self.encoder(x)
        z_future_pred = self.predictor(z_past[:, -1, :])
        z_future_pred = z_future_pred.view(-1, self.pred_steps, self.input_dim)
        z_past_reg = self.reg_proj(z_past[:, -1, :])
        return z_past, z_future_pred, z_past_reg

class MarketDataset(Dataset):
    def __init__(self, feature_df, seq_len, pred_steps):
        self.feature_df = feature_df
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.data = feature_df.values

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_steps + 1

    def __getitem__(self, idx):
        past = self.data[idx:idx + self.seq_len]
        future = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_steps]
        return torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)

class RealTimeFeatureBuffer:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.buffer = []

    def add_feature(self, feature_dict):
        feature_vector = [feature_dict[key] for key in sorted(feature_dict.keys()) if key not in ['close_price', 'timestamp']]
        self.buffer.append(feature_vector)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
        logger.info(f"Feature buffer size: {len(self.buffer)}/{self.seq_len}")

    def get_current_state(self):
        if len(self.buffer) < self.seq_len:
            logger.info("Buffer not full, waiting for more features")
            return None
        return torch.tensor(self.buffer, dtype=torch.float32)

class MPCModule:
    def __init__(self, model, horizon, action_space, cost_weights):
        self.model = model
        self.horizon = horizon
        self.action_space = action_space
        self.cost_weights = cost_weights

    def simulate_future(self, current_state, actions):
        self.model.eval()
        with torch.no_grad():
            state = current_state.clone()
            future_states = []
            for action in actions:
                z_past, z_future_pred, _ = self.model(state.unsqueeze(0))
                next_state = z_future_pred[:, 0, :]
                future_states.append(next_state)
                state = torch.cat([state[1:], next_state], dim=0)
            return torch.stack(future_states, dim=1)

    def compute_cost(self, future_states, actions):
        transaction_cost = torch.tensor([0.001 if a != 0 else 0 for a in actions]).sum()
        risk = future_states.var(dim=1).mean()
        returns = future_states[:, :, 0].mean()
        cost = (self.cost_weights['transaction'] * transaction_cost +
                self.cost_weights['risk'] * risk -
                self.cost_weights['return'] * returns)
        return cost

    def optimize_action(self, current_state):
        best_cost = float('inf')
        best_actions = None
        for action_seq in np.random.choice(self.action_space, size=(50, self.horizon)):
            future_states = self.simulate_future(current_state, action_seq)
            cost = self.compute_cost(future_states, action_seq)
            if cost < best_cost:
                best_cost = cost
                best_actions = action_seq
        return best_actions[0] if best_actions is not None else 0

# Training and Utility Functions
def compute_historical_averages(symbol, days=30):
    client = Client()
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, f"{days} days ago UTC")
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base',
                                       'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    ADV_minute = df['volume'].mean()
    ADVOL_minute = ((df['high'] - df['low']) / df['close']).mean()
    return ADV_minute, ADVOL_minute

def compute_features_from_klines(klines_df, ADV_minute, ADVOL_minute):
    df = klines_df.copy()
    df['log_return'] = np.log(df['close'] / df['open']).fillna(0)
    df['scaled_volume'] = df['volume'] / ADV_minute
    buy_volume = df['taker_buy_base']
    sell_volume = df['volume'] - buy_volume
    total_volume = df['volume']
    df['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
    df['order_flow_imbalance'] = df['order_flow_imbalance'].fillna(0)
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['scaled_volatility'] = df['volatility'] / ADVOL_minute
    df['num_trades'] = np.log1p(df['number_of_trades'])
    df['hour'] = df.index.hour / 23.0
    df['day'] = df.index.dayofweek / 6.0
    df['rsi'] = compute_rsi(df['close'], 14).fillna(50.0) / 100.0
    df['macd'] = compute_macd(df['close']).fillna(0.0)
    df['bollinger_width'] = compute_bollinger_width(df['close']).fillna(0.0)
    df['rolling_volatility'] = compute_rolling_volatility(df['close']).fillna(0.0)
    feature_cols = ['log_return', 'scaled_volume', 'order_flow_imbalance', 'scaled_volatility', 'num_trades', 'hour', 'day', 'rsi', 'macd', 'bollinger_width', 'rolling_volatility']
    return df[feature_cols]

def train_jepa(model, dataloader, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_past, x_future in dataloader:
            x_past, x_future = x_past.to(device), x_future.to(device)
            optimizer.zero_grad()
            z_past, z_future_pred, z_past_reg = model(x_past)
            pred_loss = nn.MSELoss()(z_future_pred, x_future)
            reg_loss = vicreg_loss(z_past_reg, z_future_pred[:, 0, :])
            loss = pred_loss + 0.1 * reg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

def train_model(symbol, days=90):
    client = Client()
    klines_file = f"{symbol}_klines.pkl"
    if os.path.exists(klines_file):
        logger.info(f"Loading cached klines from {klines_file}")
        with open(klines_file, 'rb') as f:
            klines = pickle.load(f)
    else:
        logger.info(f"Fetching {days} days of klines for {symbol}")
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, f"{days} days ago UTC")
        with open(klines_file, 'wb') as f:
            pickle.dump(klines, f)

    klines_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                              'quote_asset_volume', 'number_of_trades', 'taker_buy_base',
                                              'taker_buy_quote', 'ignore'])
    klines_df['timestamp'] = pd.to_datetime(klines_df['timestamp'], unit='ms')
    klines_df.set_index('timestamp', inplace=True)
    klines_df = klines_df.astype(float)

    ADV_minute, ADVOL_minute = compute_historical_averages(symbol, days)
    feature_df = compute_features_from_klines(klines_df, ADV_minute, ADVOL_minute)
    scaler = StandardScaler()
    feature_df_normalized = pd.DataFrame(
        scaler.fit_transform(feature_df),
        columns=feature_df.columns,
        index=feature_df.index
    )
    joblib.dump(scaler, 'scaler.pkl')
    seq_len = 60
    pred_steps = 5
    dataset = MarketDataset(feature_df_normalized, seq_len, pred_steps)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    input_dim = feature_df.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPAModel(input_dim, d_model=128, nhead=8, num_layers=6, pred_steps=pred_steps).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    train_jepa(model, dataloader, optimizer, device)
    torch.save(model.state_dict(), 'jepa_model.pth')
    return model, ADV_minute, ADVOL_minute, scaler

# Main Execution
def main():
    symbol = input("Enter the ticker symbol (e.g., BTCUSDT): ").strip()
    model_file = 'jepa_model.pth'
    scaler_file = 'scaler.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 11
    seq_len = 60
    pred_steps = 5

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        logger.info("No trained model or scaler found. Starting training...")
        try:
            model, ADV_minute, ADVOL_minute, scaler = train_model(symbol)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError("Cannot proceed without a trained model.")
    else:
        logger.info("Loading existing trained model and scaler...")
        ADV_minute, ADVOL_minute = compute_historical_averages(symbol)
        model = JEPAModel(input_dim, d_model=128, nhead=8, num_layers=6, pred_steps=pred_steps).to(device)
        model.load_state_dict(torch.load(model_file))
        scaler = joblib.load(scaler_file)

    model.eval()
    cost_weights = {'transaction': 0.5, 'risk': 1.0, 'return': 3.0}
    mpc = MPCModule(model, horizon=30, action_space=[0, 1, 2], cost_weights=cost_weights)
    feature_buffer = RealTimeFeatureBuffer(seq_len)
    action_logger = TradingActionLogger()

    def on_new_features(features):
        feature_buffer.add_feature(features)
        current_state = feature_buffer.get_current_state()
        if current_state is not None:
            action = mpc.optimize_action(current_state.to(device))
            action_logger.log_action(action, features, features.get('timestamp'))

    processor = RealTimeProcessor(symbol, ADV_minute, ADVOL_minute, scaler, callback=on_new_features)
    client = BinanceWebSocketClient(symbol, processor)
    client.connect()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        client.stop()

if __name__ == "__main__":
    main()