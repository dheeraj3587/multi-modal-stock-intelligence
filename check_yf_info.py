import yfinance as yf
import json

def check_yf():
    ticker = yf.Ticker("RELIANCE.NS")
    info = ticker.info
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    check_yf()
