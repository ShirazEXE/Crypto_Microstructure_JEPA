# 🪙 Crypto Trading Pipeline

A high-frequency cryptocurrency trading pipeline that collects real-time market data using WebSockets, performs feature engineering, and predicts price movements using an Self-Supervised, Energy-Based(EBM) machine learning model with Joint Embedding Predictive Architecture(JEPA) and Model Predictive Control (MPC). This project includes a web interface and a FastAPI backend for serving predictions and interacting with the system.

---

## 📂 Project Structure

```
.
├── crypto_trading_pipeline.py   # Main trading pipeline: data collection, processing, and prediction
├── main.py                      # FastAPI backend for serving model predictions
├── index.html                   # Simple frontend UI for interaction
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Features

* 📡 Real-time WebSocket data collection from Binance
* 📊 Tick-level market data processing with feature extraction
* 🤖 JEPA-based model for price prediction
* 🖥️ FastAPI backend to expose prediction endpoints
* 🌐 Web-based frontend interface (HTML)
* 📓 Efficient data storage and model persistence using `joblib`

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/crypto-trading-pipeline.git
cd crypto-trading-pipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the backend server

```bash
uvicorn main:app --reload
```

### 4. Open the frontend

Open `index.html` in your browser to interact with the system.

---

## 🧠 Model Overview

The model is built using PyTorch and implements a Long Short-Term Memory (JEPA) architecture to predict short-term price movements based on historical market features like:

* Price changes
* Order book deltas
* Trade volumes

---

## 🛠️ API Endpoints

| Method | Endpoint   | Description                       |
| ------ | ---------- | --------------------------------- |
| GET    | `/`        | Returns HTML interface            |
| POST   | `/predict` | Returns price movement prediction |

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full list of dependencies, including:

* `websocket-client`
* `python-binance`
* `torch`
* `pandas`, `numpy`
* `fastapi`, `uvicorn`

---

## 📌 Notes

* Ensure you have access to Binance API if using secured endpoints.
* The model is currently trained on simulated or historic data and may require fine-tuning for live trading.
* For actual trading integration, risk management modules should be added.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
