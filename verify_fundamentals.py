import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.market_service import market_service

def test_fundamentals():
    symbol = "RELIANCE"
    print(f"Testing YFinance Fallback for {symbol}...")
    
    # It communicates with YFinance synchronously
    data = market_service.get_fundamental_info(symbol)
    
    if data:
        print("Success!")
        print(json.dumps(data, indent=2))
        
        # Verify fields required by frontend
        required = ["company_name", "about", "ratios", "pros", "cons"]
        missing = [f for f in required if f not in data]
        if missing:
            print(f"WARNING: Missing fields: {missing}")
        else:
            print("All frontend fields present.")
    else:
        print("Failed: No data returned from YFinance fallback.")

if __name__ == "__main__":
    test_fundamentals()
