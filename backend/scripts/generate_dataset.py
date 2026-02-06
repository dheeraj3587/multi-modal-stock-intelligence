
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

# List of Nifty 100 + Midcap 50 candidates (approx 150)
SYMBOLS = [
    # Nifty 100
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ADANITOTALGAS", 
    "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "DMART", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", 
    "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA", "BEL", "BPCL", "BHARTIARTL", "BOSCHLTD", 
    "BRITANNIA", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "COLPAL", "DIVISLAB", "DLF", 
    "DRREDDY", "EICHERMOT", "GAIL", "GODREJCP", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", 
    "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDUNILVR", "ICICIBANK", "ICICIGI", 
    "ICICIPRULI", "IOC", "IRFC", "IRCTC", "INDUSINDBK", "NAUKRI", "INFY", "INDIGO", "ITC", 
    "JIOFIN", "JINDALSTEL", "JSWSTEEL", "KOTAKBANK", "LT", "LICI", "LTIM", "M&M", "MARICO", 
    "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC", "PIDILITIND", "PFC", "POWERGRID", 
    "PNB", "RECLTD", "RELIANCE", "MOTHERSON", "SBICARD", "SBILIFE", "SHRIRAMFIN", "SIEMENS", 
    "SRF", "SBIN", "SUNPHARMA", "TATACONSUM", "TCS", "TATAMOTORS", "TATAPOWER", "TATASTEEL", 
    "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR", "ULTRACEMCO", "VBL", "VEDL", "WIPRO", 
    "ZYDUSLIFE", "ZOMATO",
    # Midcap 50 (Top candidates)
    "SUZLON", "INDHOTEL", "CUMMINSIND", "YESBANK", "BHARATFORG", "HDFCAMC", "BHEL", "LUPIN", 
    "TIINDIA", "ABCAPITAL", "ACC", "ASHOKLEY", "AUROPHARMA", "CONCOR", "CGPOWER", "FEDERALBNK", 
    "GMRINFRA", "GODREJPROP", "HINDPETRO", "IDEA", "MPHASIS", "MRF", "NMDC", "OBEROIRLTY", 
    "OFSS", "PERSISTENT", "PETRONET", "PHOENIXLTD", "SAIL", "SUNDARMFIN", "SUPREMEIND", 
    "TATACOMM", "UPL", "VOLTAS", "INDUSTOWER", "L&TFH", "MUTHOOTFIN", "PIIND", "ASTRAL", 
    "APLAPOLLO", "IDFCFIRSTB", "ALKEM", "AUBANK", "DIXON", "POLYCAB", "KPITTECH", "PBFINTECH",
    "JSWENERGY", "JSWINFRA", "PRESTIGE", "LINDEINDIA", "POLICYBZR", "UNIONBANK", "IOB", "MAHABANK",
    "GICRE", "NEWINDIA", "NHPC", "OIL", "SJVN", "RVNL", "MAZAGON"
]

def fetch_stock_data():
    data = []
    
    # Process in batches to handle rate limits better
    batch_size = 20
    unique_symbols = sorted(list(set(SYMBOLS)))
    print(f"Total unique symbols to process: {len(unique_symbols)}")
    
    for i in range(0, len(unique_symbols), batch_size):
        batch = unique_symbols[i:i+batch_size]
        tickers = [f"{sym}.NS" for sym in batch]
        
        print(f"Fetching batch {i//batch_size + 1}: {tickers[0]} to {tickers[-1]}")
        
        try:
            # Use Tickers object for batch fetching info is not supported efficiently in yfinance
            # iterating is safer for 'info'
            for ticker_symbol in tqdm(tickers):
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    info = ticker.info
                    
                    market_cap = info.get('marketCap', 0)
                    if market_cap is None: market_cap = 0
                    
                    # Skip if no market cap
                    if market_cap == 0:
                        continue

                    # Determine sector
                    sector = info.get('sector', 'Unknown')
                    if sector == 'Unknown':
                         sector = info.get('industry', 'Unknown')

                    row = {
                        "symbol": ticker_symbol,
                        "company_name": info.get('longName', ticker_symbol),
                        "sector": sector,
                        "raw_market_cap": market_cap,
                        "currency": info.get('currency', 'INR')
                    }
                    data.append(row)
                    
                except Exception as e:
                    print(f"Error fetching {ticker_symbol}: {e}")
                
        except Exception as e:
            print(f"Batch failed: {e}")
            
    return data

def format_market_cap(val_inr):
    # Convert to Trilliions or Crores
    # 1 Trillion = 1,00,000 Crores = 10^12
    # 1 Lakh Crore = 1 Trillion
    
    if val_inr >= 1e12:
        val = val_inr / 1e12
        return f"₹{val:.2f}T"
    elif val_inr >= 1e10: # 1000 Crores is 10 Billion
        val = val_inr / 1e10  # This logic is non-standard for international. 
        # Standard Indian notation:
        # Market cap is usually in Crores.
        # Let's stick to user example: ₹18.88T (Trillions)
        return f"₹{val_inr/1e12:.2f}T"
    else:
        # Billions
        return f"₹{val_inr/1e9:.2f}B"

def main():
    print("Starting data collection...")
    stock_data = fetch_stock_data()
    
    # Sort by market cap descending
    stock_data.sort(key=lambda x: x['raw_market_cap'], reverse=True)
    
    # Take top 120
    top_120 = stock_data[:120]
    
    # Format output
    final_rows = []
    for rank, item in enumerate(top_120, 1):
        formatted_mc = format_market_cap(item['raw_market_cap'])
        
        final_rows.append({
            "symbol": item['symbol'],
            "company_name": item['company_name'],
            "sector": item['sector'],
            "approx_market_cap": formatted_mc,
            "source_rank": rank,
            "source_url": f"https://finance.yahoo.com/quote/{item['symbol']}"
        })
        
    df = pd.DataFrame(final_rows)
    output_path = "data/ind_top_120.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved top {len(df)} stocks to {output_path}")
    
    # Print first few rows
    print(df.head())

if __name__ == "__main__":
    main()
