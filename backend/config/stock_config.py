"""
Configuration for Stock Data and Indices.
Source of truth for all supported stocks and their metadata.
"""

# Indian Stock Mapping - NIFTY 100 Components
STOCK_DATA = {
    # Energy & Oil & Gas
    'RELIANCE': {'name': 'Reliance Industries', 'isin': 'INE002A01018', 'sector': 'Energy', 'instrument_key': 'NSE_EQ|INE002A01018', 'yf_ticker': 'RELIANCE.NS'},
    'ONGC': {'name': 'Oil & Natural Gas Corp', 'isin': 'INE213A01029', 'sector': 'Energy', 'instrument_key': 'NSE_EQ|INE213A01029', 'yf_ticker': 'ONGC.NS'},
    'IOC': {'name': 'Indian Oil Corporation', 'isin': 'INE242A01010', 'sector': 'Energy', 'instrument_key': 'NSE_EQ|INE242A01010', 'yf_ticker': 'IOC.NS'},
    'BPCL': {'name': 'Bharat Petroleum', 'isin': 'INE029A01011', 'sector': 'Energy', 'instrument_key': 'NSE_EQ|INE029A01011', 'yf_ticker': 'BPCL.NS'},
    'GAIL': {'name': 'GAIL (India) Ltd', 'isin': 'INE129A01019', 'sector': 'Energy', 'instrument_key': 'NSE_EQ|INE129A01019', 'yf_ticker': 'GAIL.NS'},
    'ADANIENT': {'name': 'Adani Enterprises', 'isin': 'INE423A01024', 'sector': 'Conglomerate', 'instrument_key': 'NSE_EQ|INE423A01024', 'yf_ticker': 'ADANIENT.NS'},
    'ADANIGREEN': {'name': 'Adani Green Energy', 'isin': 'INE364U01010', 'sector': 'Energy', 'instrument_key': 'NSE_EQ|INE364U01010', 'yf_ticker': 'ADANIGREEN.NS'},
    'ADANIPORTS': {'name': 'Adani Ports & SEZ', 'isin': 'INE742F01042', 'sector': 'Infrastructure', 'instrument_key': 'NSE_EQ|INE742F01042', 'yf_ticker': 'ADANIPORTS.NS'},
    'POWERGRID': {'name': 'Power Grid Corp', 'isin': 'INE752E01010', 'sector': 'Power', 'instrument_key': 'NSE_EQ|INE752E01010', 'yf_ticker': 'POWERGRID.NS'},
    'NTPC': {'name': 'NTPC Limited', 'isin': 'INE733E01010', 'sector': 'Power', 'instrument_key': 'NSE_EQ|INE733E01010', 'yf_ticker': 'NTPC.NS'},
    'COALINDIA': {'name': 'Coal India Limited', 'isin': 'INE522F01014', 'sector': 'Mining', 'instrument_key': 'NSE_EQ|INE522F01014', 'yf_ticker': 'COALINDIA.NS'},
    
    # IT & Technology
    'TCS': {'name': 'Tata Consultancy Services', 'isin': 'INE467B01029', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE467B01029', 'yf_ticker': 'TCS.NS'},
    'INFY': {'name': 'Infosys Limited', 'isin': 'INE009A01021', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE009A01021', 'yf_ticker': 'INFY.NS'},
    'WIPRO': {'name': 'Wipro Limited', 'isin': 'INE075A01022', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE075A01022', 'yf_ticker': 'WIPRO.NS'},
    'HCLTECH': {'name': 'HCL Technologies', 'isin': 'INE860A01027', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE860A01027', 'yf_ticker': 'HCLTECH.NS'},
    'TECHM': {'name': 'Tech Mahindra', 'isin': 'INE669C01036', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE669C01036', 'yf_ticker': 'TECHM.NS'},
    'LTIM': {'name': 'LTIMindtree Ltd', 'isin': 'INE214T01019', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE214T01019', 'yf_ticker': 'LTIM.NS'},
    'MPHASIS': {'name': 'Mphasis Limited', 'isin': 'INE356A01018', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE356A01018', 'yf_ticker': 'MPHASIS.NS'},
    'COFORGE': {'name': 'Coforge Limited', 'isin': 'INE591G01017', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE591G01017', 'yf_ticker': 'COFORGE.NS'},
    'PERSISTENT': {'name': 'Persistent Systems', 'isin': 'INE262H01013', 'sector': 'IT', 'instrument_key': 'NSE_EQ|INE262H01013', 'yf_ticker': 'PERSISTENT.NS'},
    
    # Banking
    'HDFCBANK': {'name': 'HDFC Bank Ltd', 'isin': 'INE040A01034', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE040A01034', 'yf_ticker': 'HDFCBANK.NS'},
    'ICICIBANK': {'name': 'ICICI Bank Ltd', 'isin': 'INE090A01021', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE090A01021', 'yf_ticker': 'ICICIBANK.NS'},
    'SBIN': {'name': 'State Bank of India', 'isin': 'INE062A01020', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE062A01020', 'yf_ticker': 'SBIN.NS'},
    'KOTAKBANK': {'name': 'Kotak Mahindra Bank', 'isin': 'INE237A01028', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE237A01028', 'yf_ticker': 'KOTAKBANK.NS'},
    'AXISBANK': {'name': 'Axis Bank Ltd', 'isin': 'INE238A01034', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE238A01034', 'yf_ticker': 'AXISBANK.NS'},
    'INDUSINDBK': {'name': 'IndusInd Bank', 'isin': 'INE095A01012', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE095A01012', 'yf_ticker': 'INDUSINDBK.NS'},
    'BANKBARODA': {'name': 'Bank of Baroda', 'isin': 'INE028A01039', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE028A01039', 'yf_ticker': 'BANKBARODA.NS'},
    'PNB': {'name': 'Punjab National Bank', 'isin': 'INE160A01022', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE160A01022', 'yf_ticker': 'PNB.NS'},
    'CANBK': {'name': 'Canara Bank', 'isin': 'INE476A01014', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE476A01014', 'yf_ticker': 'CANBK.NS'},
    'FEDERALBNK': {'name': 'Federal Bank', 'isin': 'INE171A01029', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE171A01029', 'yf_ticker': 'FEDERALBNK.NS'},
    'IDFCFIRSTB': {'name': 'IDFC First Bank', 'isin': 'INE092T01019', 'sector': 'Banking', 'instrument_key': 'NSE_EQ|INE092T01019', 'yf_ticker': 'IDFCFIRSTB.NS'},
    
    # Finance & NBFC
    'BAJFINANCE': {'name': 'Bajaj Finance', 'isin': 'INE296A01024', 'sector': 'Finance', 'instrument_key': 'NSE_EQ|INE296A01024', 'yf_ticker': 'BAJFINANCE.NS'},
    'BAJAJFINSV': {'name': 'Bajaj Finserv', 'isin': 'INE918I01018', 'sector': 'Finance', 'instrument_key': 'NSE_EQ|INE918I01018', 'yf_ticker': 'BAJAJFINSV.NS'},
    'SBILIFE': {'name': 'SBI Life Insurance', 'isin': 'INE123W01016', 'sector': 'Insurance', 'instrument_key': 'NSE_EQ|INE123W01016', 'yf_ticker': 'SBILIFE.NS'},
    'HDFCLIFE': {'name': 'HDFC Life Insurance', 'isin': 'INE795G01014', 'sector': 'Insurance', 'instrument_key': 'NSE_EQ|INE795G01014', 'yf_ticker': 'HDFCLIFE.NS'},
    'ICICIPRULI': {'name': 'ICICI Prudential Life', 'isin': 'INE726G01019', 'sector': 'Insurance', 'instrument_key': 'NSE_EQ|INE726G01019', 'yf_ticker': 'ICICIPRULI.NS'},
    'ICICIGI': {'name': 'ICICI Lombard GIC', 'isin': 'INE765G01017', 'sector': 'Insurance', 'instrument_key': 'NSE_EQ|INE765G01017', 'yf_ticker': 'ICICIGI.NS'},
    'SBICARD': {'name': 'SBI Cards & Payment', 'isin': 'INE018E01016', 'sector': 'Finance', 'instrument_key': 'NSE_EQ|INE018E01016', 'yf_ticker': 'SBICARD.NS'},
    'CHOLAFIN': {'name': 'Cholamandalam Finance', 'isin': 'INE121A01024', 'sector': 'Finance', 'instrument_key': 'NSE_EQ|INE121A01024', 'yf_ticker': 'CHOLAFIN.NS'},
    'MUTHOOTFIN': {'name': 'Muthoot Finance', 'isin': 'INE414G01012', 'sector': 'Finance', 'instrument_key': 'NSE_EQ|INE414G01012', 'yf_ticker': 'MUTHOOTFIN.NS'},
    'PEL': {'name': 'Piramal Pharma', 'isin': 'INE140A01024', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE140A01024', 'yf_ticker': 'PPLPHARMA.NS'},
    
    # FMCG & Consumer
    'HINDUNILVR': {'name': 'Hindustan Unilever', 'isin': 'INE030A01027', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE030A01027', 'yf_ticker': 'HINDUNILVR.NS'},
    'ITC': {'name': 'ITC Limited', 'isin': 'INE154A01025', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE154A01025', 'yf_ticker': 'ITC.NS'},
    'NESTLEIND': {'name': 'Nestle India', 'isin': 'INE239A01016', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE239A01016', 'yf_ticker': 'NESTLEIND.NS'},
    'BRITANNIA': {'name': 'Britannia Industries', 'isin': 'INE216A01030', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE216A01030', 'yf_ticker': 'BRITANNIA.NS'},
    'DABUR': {'name': 'Dabur India', 'isin': 'INE016A01026', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE016A01026', 'yf_ticker': 'DABUR.NS'},
    'MARICO': {'name': 'Marico Limited', 'isin': 'INE196A01026', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE196A01026', 'yf_ticker': 'MARICO.NS'},
    'GODREJCP': {'name': 'Godrej Consumer Products', 'isin': 'INE102D01028', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE102D01028', 'yf_ticker': 'GODREJCP.NS'},
    'COLPAL': {'name': 'Colgate-Palmolive', 'isin': 'INE259A01022', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE259A01022', 'yf_ticker': 'COLPAL.NS'},
    'TATACONSUM': {'name': 'Tata Consumer Products', 'isin': 'INE192A01025', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE192A01025', 'yf_ticker': 'TATACONSUM.NS'},
    'VBL': {'name': 'Varun Beverages', 'isin': 'INE200M01013', 'sector': 'FMCG', 'instrument_key': 'NSE_EQ|INE200M01013', 'yf_ticker': 'VBL.NS'},
    'TITAN': {'name': 'Titan Company', 'isin': 'INE280A01028', 'sector': 'Consumer', 'instrument_key': 'NSE_EQ|INE280A01028', 'yf_ticker': 'TITAN.NS'},
    'ASIANPAINT': {'name': 'Asian Paints', 'isin': 'INE021A01026', 'sector': 'Consumer', 'instrument_key': 'NSE_EQ|INE021A01026', 'yf_ticker': 'ASIANPAINT.NS'},
    'PIDILITIND': {'name': 'Pidilite Industries', 'isin': 'INE318A01026', 'sector': 'Consumer', 'instrument_key': 'NSE_EQ|INE318A01026', 'yf_ticker': 'PIDILITIND.NS'},
    'BERGEPAINT': {'name': 'Berger Paints', 'isin': 'INE463A01038', 'sector': 'Consumer', 'instrument_key': 'NSE_EQ|INE463A01038', 'yf_ticker': 'BERGEPAINT.NS'},
    
    # Auto & Auto Ancillary
    'TATAMOTORS': {'name': 'Tata Motors Ltd', 'isin': 'INE155A01022', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE155A01022', 'yf_ticker': 'TATAMOTORS.NS'},
    'MARUTI': {'name': 'Maruti Suzuki', 'isin': 'INE585B01010', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE585B01010', 'yf_ticker': 'MARUTI.NS'},
    'M&M': {'name': 'Mahindra & Mahindra', 'isin': 'INE101A01026', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE101A01026', 'yf_ticker': 'M&M.NS'},
    'BAJAJ-AUTO': {'name': 'Bajaj Auto', 'isin': 'INE917I01010', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE917I01010', 'yf_ticker': 'BAJAJ-AUTO.NS'},
    'HEROMOTOCO': {'name': 'Hero MotoCorp', 'isin': 'INE158A01026', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE158A01026', 'yf_ticker': 'HEROMOTOCO.NS'},
    'EICHERMOT': {'name': 'Eicher Motors', 'isin': 'INE066A01021', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE066A01021', 'yf_ticker': 'EICHERMOT.NS'},
    'TVSMOTOR': {'name': 'TVS Motor Company', 'isin': 'INE494B01023', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE494B01023', 'yf_ticker': 'TVSMOTOR.NS'},
    'BOSCHLTD': {'name': 'Bosch Limited', 'isin': 'INE323A01026', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE323A01026', 'yf_ticker': 'BOSCHLTD.NS'},
    'MOTHERSON': {'name': 'Motherson Sumi', 'isin': 'INE775A01035', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE775A01035', 'yf_ticker': 'MOTHERSON.NS'},
    'BALKRISIND': {'name': 'Balkrishna Industries', 'isin': 'INE787D01026', 'sector': 'Auto', 'instrument_key': 'NSE_EQ|INE787D01026', 'yf_ticker': 'BALKRISIND.NS'},
    
    # Pharma & Healthcare
    'SUNPHARMA': {'name': 'Sun Pharmaceutical', 'isin': 'INE044A01036', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE044A01036', 'yf_ticker': 'SUNPHARMA.NS'},
    'DRREDDY': {'name': 'Dr. Reddys Labs', 'isin': 'INE089A01023', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE089A01023', 'yf_ticker': 'DRREDDY.NS'},
    'CIPLA': {'name': 'Cipla Limited', 'isin': 'INE059A01026', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE059A01026', 'yf_ticker': 'CIPLA.NS'},
    'DIVISLAB': {'name': 'Divis Laboratories', 'isin': 'INE361B01024', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE361B01024', 'yf_ticker': 'DIVISLAB.NS'},
    'APOLLOHOSP': {'name': 'Apollo Hospitals', 'isin': 'INE437A01024', 'sector': 'Healthcare', 'instrument_key': 'NSE_EQ|INE437A01024', 'yf_ticker': 'APOLLOHOSP.NS'},
    'TORNTPHARM': {'name': 'Torrent Pharma', 'isin': 'INE685A01028', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE685A01028', 'yf_ticker': 'TORNTPHARM.NS'},
    'LUPIN': {'name': 'Lupin Limited', 'isin': 'INE326A01037', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE326A01037', 'yf_ticker': 'LUPIN.NS'},
    'BIOCON': {'name': 'Biocon Limited', 'isin': 'INE376G01013', 'sector': 'Pharma', 'instrument_key': 'NSE_EQ|INE376G01013', 'yf_ticker': 'BIOCON.NS'},
    'MAXHEALTH': {'name': 'Max Healthcare', 'isin': 'INE027H01010', 'sector': 'Healthcare', 'instrument_key': 'NSE_EQ|INE027H01010', 'yf_ticker': 'MAXHEALTH.NS'},
    
    # Infrastructure & Engineering
    'LT': {'name': 'Larsen & Toubro', 'isin': 'INE018A01030', 'sector': 'Engineering', 'instrument_key': 'NSE_EQ|INE018A01030', 'yf_ticker': 'LT.NS'},
    'ULTRACEMCO': {'name': 'UltraTech Cement', 'isin': 'INE481G01011', 'sector': 'Cement', 'instrument_key': 'NSE_EQ|INE481G01011', 'yf_ticker': 'ULTRACEMCO.NS'},
    'GRASIM': {'name': 'Grasim Industries', 'isin': 'INE047A01021', 'sector': 'Cement', 'instrument_key': 'NSE_EQ|INE047A01021', 'yf_ticker': 'GRASIM.NS'},
    'SHREECEM': {'name': 'Shree Cement', 'isin': 'INE070A01015', 'sector': 'Cement', 'instrument_key': 'NSE_EQ|INE070A01015', 'yf_ticker': 'SHREECEM.NS'},
    'AMBUJACEM': {'name': 'Ambuja Cements', 'isin': 'INE079A01024', 'sector': 'Cement', 'instrument_key': 'NSE_EQ|INE079A01024', 'yf_ticker': 'AMBUJACEM.NS'},
    'ACC': {'name': 'ACC Limited', 'isin': 'INE012A01025', 'sector': 'Cement', 'instrument_key': 'NSE_EQ|INE012A01025', 'yf_ticker': 'ACC.NS'},
    'DLF': {'name': 'DLF Limited', 'isin': 'INE271C01023', 'sector': 'Realty', 'instrument_key': 'NSE_EQ|INE271C01023', 'yf_ticker': 'DLF.NS'},
    'GODREJPROP': {'name': 'Godrej Properties', 'isin': 'INE484J01027', 'sector': 'Realty', 'instrument_key': 'NSE_EQ|INE484J01027', 'yf_ticker': 'GODREJPROP.NS'},
    
    # Metals & Mining
    'TATASTEEL': {'name': 'Tata Steel', 'isin': 'INE081A01020', 'sector': 'Metals', 'instrument_key': 'NSE_EQ|INE081A01020', 'yf_ticker': 'TATASTEEL.NS'},
    'JSWSTEEL': {'name': 'JSW Steel', 'isin': 'INE019A01038', 'sector': 'Metals', 'instrument_key': 'NSE_EQ|INE019A01038', 'yf_ticker': 'JSWSTEEL.NS'},
    'HINDALCO': {'name': 'Hindalco Industries', 'isin': 'INE038A01020', 'sector': 'Metals', 'instrument_key': 'NSE_EQ|INE038A01020', 'yf_ticker': 'HINDALCO.NS'},
    'VEDL': {'name': 'Vedanta Limited', 'isin': 'INE205A01025', 'sector': 'Metals', 'instrument_key': 'NSE_EQ|INE205A01025', 'yf_ticker': 'VEDL.NS'},
    'SAIL': {'name': 'Steel Authority of India', 'isin': 'INE114A01011', 'sector': 'Metals', 'instrument_key': 'NSE_EQ|INE114A01011', 'yf_ticker': 'SAIL.NS'},
    'NMDC': {'name': 'NMDC Limited', 'isin': 'INE584A01023', 'sector': 'Mining', 'instrument_key': 'NSE_EQ|INE584A01023', 'yf_ticker': 'NMDC.NS'},
    
    # Telecom & Media
    'BHARTIARTL': {'name': 'Bharti Airtel', 'isin': 'INE397D01024', 'sector': 'Telecom', 'instrument_key': 'NSE_EQ|INE397D01024', 'yf_ticker': 'BHARTIARTL.NS'},
    'IDEA': {'name': 'Vodafone Idea', 'isin': 'INE669E01016', 'sector': 'Telecom', 'instrument_key': 'NSE_EQ|INE669E01016', 'yf_ticker': 'IDEA.NS'},
    'TATACOMM': {'name': 'Tata Communications', 'isin': 'INE151A01013', 'sector': 'Telecom', 'instrument_key': 'NSE_EQ|INE151A01013', 'yf_ticker': 'TATACOMM.NS'},
    'ZEEL': {'name': 'Zee Entertainment', 'isin': 'INE256A01028', 'sector': 'Media', 'instrument_key': 'NSE_EQ|INE256A01028', 'yf_ticker': 'ZEEL.NS'},
    'PVR': {'name': 'PVR INOX Ltd', 'isin': 'INE191H01014', 'sector': 'Media', 'instrument_key': 'NSE_EQ|INE191H01014', 'yf_ticker': 'PVRINOX.NS'},
    
    # Retail & E-commerce
    'DMART': {'name': 'Avenue Supermarts', 'isin': 'INE192R01011', 'sector': 'Retail', 'instrument_key': 'NSE_EQ|INE192R01011', 'yf_ticker': 'DMART.NS'},
    'TRENT': {'name': 'Trent Limited', 'isin': 'INE849A01020', 'sector': 'Retail', 'instrument_key': 'NSE_EQ|INE849A01020', 'yf_ticker': 'TRENT.NS'},
    'NYKAA': {'name': 'FSN E-Commerce (Nykaa)', 'isin': 'INE388Y01029', 'sector': 'E-commerce', 'instrument_key': 'NSE_EQ|INE388Y01029', 'yf_ticker': 'NYKAA.NS'},
    'ZOMATO': {'name': 'Zomato (Eternal)', 'isin': 'INE758T01015', 'sector': 'E-commerce', 'instrument_key': 'NSE_EQ|INE758T01015', 'yf_ticker': 'ZOMATO.NS'},
    'PAYTM': {'name': 'One97 Communications', 'isin': 'INE982J01020', 'sector': 'Fintech', 'instrument_key': 'NSE_EQ|INE982J01020', 'yf_ticker': 'PAYTM.NS'},
    'POLICYBZR': {'name': 'PB Fintech', 'isin': 'INE417T01026', 'sector': 'Fintech', 'instrument_key': 'NSE_EQ|INE417T01026', 'yf_ticker': 'POLICYBZR.NS'},
    
    # Chemicals & Fertilizers
    'UPL': {'name': 'UPL Limited', 'isin': 'INE628A01036', 'sector': 'Chemicals', 'instrument_key': 'NSE_EQ|INE628A01036', 'yf_ticker': 'UPL.NS'},
    'SRF': {'name': 'SRF Limited', 'isin': 'INE647A01010', 'sector': 'Chemicals', 'instrument_key': 'NSE_EQ|INE647A01010', 'yf_ticker': 'SRF.NS'},
    'PIIND': {'name': 'PI Industries', 'isin': 'INE603J01018', 'sector': 'Chemicals', 'instrument_key': 'NSE_EQ|INE603J01018', 'yf_ticker': 'PIIND.NS'},
    'ATUL': {'name': 'Atul Limited', 'isin': 'INE100A01010', 'sector': 'Chemicals', 'instrument_key': 'NSE_EQ|INE100A01010', 'yf_ticker': 'ATUL.NS'},
    
    # Others
    'INDIGO': {'name': 'InterGlobe Aviation', 'isin': 'INE646L01027', 'sector': 'Aviation', 'instrument_key': 'NSE_EQ|INE646L01027', 'yf_ticker': 'INDIGO.NS'},
    'HAVELLS': {'name': 'Havells India', 'isin': 'INE176B01034', 'sector': 'Electricals', 'instrument_key': 'NSE_EQ|INE176B01034', 'yf_ticker': 'HAVELLS.NS'},
    'SIEMENS': {'name': 'Siemens Limited', 'isin': 'INE003A01024', 'sector': 'Engineering', 'instrument_key': 'NSE_EQ|INE003A01024', 'yf_ticker': 'SIEMENS.NS'},
    'ABB': {'name': 'ABB India', 'isin': 'INE117A01022', 'sector': 'Engineering', 'instrument_key': 'NSE_EQ|INE117A01022', 'yf_ticker': 'ABB.NS'},
    'BEL': {'name': 'Bharat Electronics', 'isin': 'INE263A01024', 'sector': 'Defence', 'instrument_key': 'NSE_EQ|INE263A01024', 'yf_ticker': 'BEL.NS'},
    'HAL': {'name': 'Hindustan Aeronautics', 'isin': 'INE066F01020', 'sector': 'Defence', 'instrument_key': 'NSE_EQ|INE066F01020', 'yf_ticker': 'HAL.NS'},
    'IRCTC': {'name': 'IRCTC Ltd', 'isin': 'INE335Y01020', 'sector': 'Travel', 'instrument_key': 'NSE_EQ|INE335Y01020', 'yf_ticker': 'IRCTC.NS'},
    'INDIANHOTELS': {'name': 'Indian Hotels', 'isin': 'INE053A01029', 'sector': 'Hotels', 'instrument_key': 'NSE_EQ|INE053A01029', 'yf_ticker': 'INDHOTEL.NS'},
    'PAGEIND': {'name': 'Page Industries', 'isin': 'INE761H01022', 'sector': 'Textiles', 'instrument_key': 'NSE_EQ|INE761H01022', 'yf_ticker': 'PAGEIND.NS'},
    'TATAPOWER': {'name': 'Tata Power', 'isin': 'INE245A01021', 'sector': 'Power', 'instrument_key': 'NSE_EQ|INE245A01021', 'yf_ticker': 'TATAPOWER.NS'},
    'ADANIPOWER': {'name': 'Adani Power', 'isin': 'INE814H01011', 'sector': 'Power', 'instrument_key': 'NSE_EQ|INE814H01011', 'yf_ticker': 'ADANIPOWER.NS'},
    'JSWENERGY': {'name': 'JSW Energy', 'isin': 'INE121E01018', 'sector': 'Power', 'instrument_key': 'NSE_EQ|INE121E01018', 'yf_ticker': 'JSWENERGY.NS'},
}

# Index mapping
INDEX_DATA = {
    'NIFTY50': {'name': 'NIFTY 50', 'instrument_key': 'NSE_INDEX|Nifty 50', 'yf_ticker': '^NSEI'},
    'SENSEX': {'name': 'SENSEX', 'instrument_key': 'BSE_INDEX|SENSEX', 'yf_ticker': '^BSESN'},
    'BANKNIFTY': {'name': 'BANK NIFTY', 'instrument_key': 'NSE_INDEX|Nifty Bank', 'yf_ticker': '^NSEBANK'},
}
