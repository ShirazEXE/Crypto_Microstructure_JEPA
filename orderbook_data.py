from decouple import config
from binance import ThreadedWebsocketManager
import psycopg2
from psycopg2.extras import Json
import datetime
import logging

# Load API keys
api_key = config("API_KEY")
api_secret = config("SECRET_KEY")
db_pass = config("DB_PASS")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Database connection
    connection = psycopg2.connect(
        user="postgres",
        password=db_pass,
        host="127.0.0.1",
        port="5432",
        database="postgres"
    )

    # Level 2 order book streams (20 depth levels)
    streams = ["btcusdt@depth20", "ethusdt@depth20"]

    # Start WebSocket Manager
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()

    def handle_message(msg):
        """Handles incoming WebSocket order book messages."""
        try:
            data = msg.get("data", msg)

            if "E" in data:  # Incremental order book update
                timestamp = datetime.datetime.fromtimestamp(data["E"] / 1000)
                symbol = data["s"].lower()
                bids = data["b"]
                asks = data["a"]
                logging.info(f"Update: {symbol} | Bids: {len(bids)} | Asks: {len(asks)}")

            elif "lastUpdateId" in data:  # Initial order book snapshot
                timestamp = datetime.datetime.utcnow()  # No timestamp in snapshot, use system time
                symbol = streams[0].split("@")[0]  # Extract symbol from stream name
                bids = data["bids"]
                asks = data["asks"]
                logging.info(f"Snapshot: {symbol} | Bids: {len(bids)} | Asks: {len(asks)}")

            else:
                logging.warning(f"Unexpected message format: {data}")
                return

            # Insert order book data into database
            with connection.cursor() as cursor:
                query = """INSERT INTO orderbook_data (time, symbol, side, price, quantity)
                        VALUES (%s, %s, %s, %s, %s)"""

                # Insert bids
                for bid in bids:
                    cursor.execute(query, (timestamp, symbol, 'bid', float(bid[0]), float(bid[1])))

                # Insert asks
                for ask in asks:
                    cursor.execute(query, (timestamp, symbol, 'ask', float(ask[0]), float(ask[1])))

                connection.commit()
                
        except Exception as e:
            logging.error(f"Error processing message: {msg} | Error: {e}")
            connection.rollback()

    # Start multiplexed WebSocket
    twm.start_multiplex_socket(callback=handle_message, streams=streams)

    # Keep script running
    try:
        twm.join()
    except KeyboardInterrupt:
        logging.info("Shutting down WebSocket...")
        twm.stop()
        connection.close()

if __name__ == "__main__":
    main()
