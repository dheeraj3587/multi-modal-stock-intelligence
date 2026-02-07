import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.screener_service import ScreenerService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_screener():
    service = ScreenerService()
    symbol = "RELIANCE"
    print(f"Testing ScreenerService for {symbol}...")
    try:
        data = service.get_company_details(symbol, "Reliance Industries")
        if data:
            print("Success!")
            print(f"Company: {data.get('company_name')}")
            print(f"Ratios: {data.get('ratios')}")
        else:
            print("Failed: No data returned.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_screener()
