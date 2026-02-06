import yfinance as yf

def test_ticker():
    symbol = "TCS.NS"
    print(f"Testing {symbol}...")
    t = yf.Ticker(symbol)
    
    print("Testing .fast_info...")
    try:
        fast = t.fast_info
        print(f"Price: {fast.last_price}")
        print(f"Previous Close: {fast.previous_close}")
    except Exception as e:
        print(f"fast_info failed: {e}")

    print("\nTesting .info...")
    try:
        info = t.info
        print(f"Price from info: {info.get('regularMarketPrice')}")
    except Exception as e:
        print(f"info failed: {e}")

    print("\nTesting .history...")
    try:
        hist = t.history(period="1d")
        print(hist)
    except Exception as e:
        print(f"history failed: {e}")

if __name__ == "__main__":
    test_ticker()
