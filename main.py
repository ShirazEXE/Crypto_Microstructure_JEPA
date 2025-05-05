from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import asyncio
import pandas as pd
import os
import logging
import numpy as np
import torch
import joblib
from datetime import datetime
import math
from typing import List, Dict, Any
import nest_asyncio

# Custom modules
from crypto_trading_pipeline import (
    compute_rsi, compute_macd, compute_bollinger_width, compute_rolling_volatility,
    BinanceWebSocketClient, RealTimeProcessor, JEPAModel, MPCModule, RealTimeFeatureBuffer,
    TradingActionLogger
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nest_asyncio.apply()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

connected_clients = []
symbol = "BTCUSDT"
processor = None
binance_client = None
mpc = None
feature_buffer = None
action_logger = None
model = None
device = None
price_history = []
trading_actions = []
model_predictions = []
initialized = False

MODEL_FILE = 'jepa_model.pth'
SCALER_FILE = 'scaler.pkl'


def sanitize_for_json(data):
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data


def initialize_trading_system(selected_symbol: str = "BTCUSDT"):
    global symbol, processor, binance_client, mpc, feature_buffer, action_logger, model, device, initialized
    try:
        symbol = selected_symbol
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
            logger.error("Model files not found. Please run crypto_trading_pipeline.py first to train the model.")
            return False

        input_dim = 11
        seq_len = 10
        pred_steps = 5

        model = JEPAModel(input_dim, d_model=128, nhead=8, num_layers=6, pred_steps=pred_steps).to(device)
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()

        scaler = joblib.load(SCALER_FILE)

        ADV_minute = 100.0
        ADVOL_minute = 0.01

        cost_weights = {'transaction': 0.5, 'risk': 1.0, 'return': 3.0}
        mpc = MPCModule(model, horizon=30, action_space=[0, 1, 2], cost_weights=cost_weights)

        feature_buffer = RealTimeFeatureBuffer(seq_len)
        action_logger = TradingActionLogger()

        def on_new_features(features):
            feature_buffer.add_feature(features)
            current_state = feature_buffer.get_current_state()

            price_history.append({
                'timestamp': features['timestamp'].isoformat() if hasattr(features['timestamp'], 'isoformat') else str(features['timestamp']),
                'price': features['close_price']
            })
            if len(price_history) > 1000:
                price_history.pop(0)

            if current_state is not None:
                action = mpc.optimize_action(current_state.to(device))
                action_desc = {0: 'hold', 1: 'buy', 2: 'sell'}[action]

                action_logger.log_action(action, features, features.get('timestamp'))
                trading_actions.append({
                    'timestamp': features['timestamp'].isoformat(),
                    'action': action_desc,
                    'price': features['close_price'],
                    'rsi': features['rsi'] * 100,
                    'macd': features['macd'],
                    'reason': action_logger.get_reason(action, features)
                })
                if len(trading_actions) > 100:
                    trading_actions.pop(0)

                with torch.no_grad():
                    _, future_pred, _ = model(current_state.unsqueeze(0).to(device))
                    future_prices = [float(features['close_price'])]
                    for i in range(pred_steps):
                        log_return = future_pred[0, i, 0].item()
                        # Constrain log return to ±0.01 per minute for realistic price changes
                        log_return = max(min(log_return, 0.01), -0.01)
                        future_prices.append(future_prices[-1] * np.exp(log_return))

                    model_predictions.append({
                        'timestamp': features['timestamp'].isoformat(),
                        'current_price': features['close_price'],
                        'predicted_prices': future_prices[1:]
                    })
                    if len(model_predictions) > 100:
                        model_predictions.pop(0)

        processor = RealTimeProcessor(symbol, ADV_minute, ADVOL_minute, scaler, callback=on_new_features)
        binance_client = BinanceWebSocketClient(symbol, processor)
        binance_client.connect()

        initialized = True
        logger.info(f"Trading system initialized for {symbol}")
        return True

    except Exception as e:
        logger.error(f"Error initializing trading system: {str(e)}")
        return False


async def broadcast_updates():
    if not connected_clients:
        return

    data = {
        'price_history': price_history[-100:],
        'trading_actions': trading_actions[-20:],
        'model_predictions': model_predictions[-1] if model_predictions else None,
    }

    if price_history:
        prices = pd.Series([p['price'] for p in price_history[-100:]])
        data['indicators'] = {
            'rsi': compute_rsi(prices).tolist() if len(prices) >= 14 else [],
            'macd': compute_macd(prices).tolist() if len(prices) >= 26 else [],
            'bollinger_width': compute_bollinger_width(prices).tolist() if len(prices) >= 20 else [],
        }

    for client in connected_clients:
        try:
            await client.send_json(sanitize_for_json(data))
        except Exception as e:
            logger.error(f"Error sending data to client: {str(e)}")


async def periodic_broadcast_task():
    while True:
        await asyncio.sleep(60)  # 1-minute interval
        await broadcast_updates()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        if not initialized:
            initialize_trading_system()

        data = {
            'price_history': price_history[-100:],
            'trading_actions': trading_actions[-20:],
            'model_predictions': model_predictions[-1] if model_predictions else None,
            'initialized': initialized
        }

        if price_history:
            prices = pd.Series([p['price'] for p in price_history[-100:]])
            data['indicators'] = {
                'rsi': compute_rsi(prices).tolist() if len(prices) >= 14 else [],
                'macd': compute_macd(prices).tolist() if len(prices) >= 26 else [],
                'bollinger_width': compute_bollinger_width(prices).tolist() if len(prices) >= 20 else [],
            }

        await websocket.send_json(sanitize_for_json(data))

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get('action') == 'change_symbol':
                new_symbol = data.get('symbol', symbol)
                logger.info(f"Changing symbol to {new_symbol}")
                if binance_client:
                    binance_client.stop()
                success = initialize_trading_system(new_symbol)
                await websocket.send_json({'action': 'symbol_changed', 'success': success, 'symbol': new_symbol})

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@app.get("/")
async def get_index():
    return FileResponse('index.html')


@app.on_event("startup")
async def startup_event():
    initialize_trading_system()
    asyncio.create_task(periodic_broadcast_task())


@app.on_event("shutdown")
async def shutdown_event():
    global binance_client
    if binance_client:
        binance_client.stop()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)