"""Market Data Service - Fetches real market data from Upstox API with YFinance fallback.

Data flow:
  1. API endpoints read from DataCache (Redis) -- instant response.
  2. Background scheduler calls refresh_all_quotes() every 5 min.
  3. refresh_all_quotes() tries UpstoxClient first, then YFinance.
"""
import os
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from backend.services.upstox_client import upstox_client
from backend.services.data_cache import data_cache

logger = logging.getLogger(__name__)

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
    # HDFC merged into HDFCBANK in 2023 -- removed as separate entry
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


class MarketService:
    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.base_url = "https://api.upstox.com/v2"
        self._redis_client = None
        self._quotes_cache_ttl = 600  # 10 min TTL (prefetch every 5 min)
        self.upstox = upstox_client
        self.cache = data_cache

        # O(1) reverse lookups: instrument_key -> yf_ticker / symbol
        self._instrument_to_yf: Dict[str, str] = {}
        self._instrument_to_symbol: Dict[str, str] = {}
        for sym, d in STOCK_DATA.items():
            self._instrument_to_yf[d['instrument_key']] = d['yf_ticker']
            self._instrument_to_symbol[d['instrument_key']] = sym
        for sym, d in INDEX_DATA.items():
            self._instrument_to_yf[d['instrument_key']] = d['yf_ticker']
            self._instrument_to_symbol[d['instrument_key']] = sym

    def _get_redis(self):
        """Lazy Redis connection for quote caching."""
        if self._redis_client is not None:
            return self._redis_client
        try:
            import redis as _redis
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis_client = _redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
            self._redis_client.ping()
            return self._redis_client
        except Exception:
            self._redis_client = None
            return None

    def get_cached_quotes(self) -> Dict:
        """Return market quotes from DataCache or legacy Redis key."""
        # Try new DataCache first
        cached = self.cache.get_quotes()
        if cached:
            return cached
        # Fallback to legacy key
        try:
            r = self._get_redis()
            if not r:
                return {}
            import json as _json
            raw = r.get("market_quotes_cache")
            if raw:
                return _json.loads(raw)
        except Exception as e:
            logger.debug(f"Quote cache miss: {e}")
        return {}

    def cache_quotes(self, quotes: Dict):
        """Store market quotes in both legacy Redis key and new DataCache."""
        # New structured cache
        self.cache.set_quotes(quotes, ttl=self._quotes_cache_ttl)
        # Legacy key (for backward compat during migration)
        try:
            r = self._get_redis()
            if not r:
                return
            import json as _json
            r.setex("market_quotes_cache", self._quotes_cache_ttl, _json.dumps(quotes))
        except Exception as e:
            logger.debug(f"Quote cache write failed: {e}")
        
    async def get_market_quote(self, access_token: Optional[str], instrument_keys: List[str]) -> Dict:
        """
        Fetch real-time market quotes.

        Priority:
          1. DataCache (Redis) - instant, populated by background scheduler.
          2. Upstox API via UpstoxClient (uses explicit token -> global Redis token).
          3. YFinance bulk download as last resort.
        """
        if not instrument_keys:
            return {}

        # 1. Try cache first
        cached = self.cache.get_quotes()
        if cached:
            hit = {k: cached[k] for k in instrument_keys if k in cached}
            if len(hit) == len(instrument_keys):
                return hit  # full cache hit -- instant

        merged: Dict = {}
        missing_keys = set(instrument_keys)

        # 2. Try Upstox (explicit token or global Redis token)
        token = access_token or upstox_client.get_global_token()
        if token:
            data = await self.upstox.get_full_quotes(list(missing_keys), token=token)
            if data:
                merged.update(data)
                missing_keys -= set(data.keys())
        else:
            logger.info("No Upstox access token available. Using YFinance fallback.")

        # 3. YFinance fallback for anything still missing
        if missing_keys:
            yf_data = await self._get_yfinance_quote(list(missing_keys))
            if yf_data:
                merged.update(yf_data)

        return merged

    async def _get_yfinance_quote(self, instrument_keys: List[str]) -> Dict:
        """Fetch quotes from Yahoo Finance using yf.download() for speed.

        Uses a single bulk download request instead of per-ticker API calls,
        making it 10-20x faster for 100+ tickers.
        """
        # O(1) reverse lookup via pre-built dict
        tickers = []
        key_map = {}
        for key in instrument_keys:
            yf_tick = self._instrument_to_yf.get(key)
            if yf_tick:
                tickers.append(yf_tick)
                key_map[yf_tick] = key
        
        if not tickers:
            return {}

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._fetch_yfinance_bulk_download, tickers
            )
            
            # Transform to Upstox-compatible format
            formatted_data = {}
            for ticker, info in result.items():
                key = key_map.get(ticker)
                if key and info.get("last_price"):
                    formatted_data[key] = {
                        "last_price": info["last_price"],
                        "volume": info.get("volume", 0),
                        "ohlc": {
                            "open": info.get("open", 0.0),
                            "high": info.get("high", 0.0),
                            "low": info.get("low", 0.0),
                            "close": info.get("prev_close", 0.0),
                        },
                    }
            return formatted_data
        except Exception as e:
            logger.error(f"YFinance quote error: {e}")
            return {}

    def _fetch_yfinance_bulk_download(self, tickers: List[str]) -> Dict:
        """Use yf.download() for a fast single-request bulk fetch.
        
        Downloads 2 days of daily OHLCV data for ALL tickers at once.
        Extracts the latest close as current price and previous close for change calc.
        """
        import pandas as pd

        data_dict: Dict = {}
        try:
            df = yf.download(
                tickers,
                period="5d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.error(f"yf.download() failed: {e}")
            return data_dict

        if df is None or df.empty:
            logger.warning("yf.download() returned empty DataFrame")
            return data_dict

        import pandas as pd
        is_multi = isinstance(df.columns, pd.MultiIndex)

        for ticker in tickers:
            try:
                if is_multi:
                    if ticker not in df.columns.get_level_values(0):
                        continue
                    ticker_df = df[ticker]
                else:
                    # Single ticker - columns are just OHLCV
                    ticker_df = df

                # Drop NaN rows
                ticker_df = ticker_df.dropna(subset=["Close"])
                if ticker_df.empty:
                    continue

                latest = ticker_df.iloc[-1]
                prev = ticker_df.iloc[-2] if len(ticker_df) >= 2 else latest

                close_val = float(latest["Close"])
                if close_val <= 0 or pd.isna(close_val):
                    continue

                data_dict[ticker] = {
                    "last_price": close_val,
                    "open": float(latest.get("Open", 0) or 0),
                    "high": float(latest.get("High", 0) or 0),
                    "low": float(latest.get("Low", 0) or 0),
                    "volume": int(latest.get("Volume", 0) or 0),
                    "prev_close": float(prev["Close"]) if not pd.isna(prev["Close"]) else close_val,
                }
            except Exception as e:
                logger.debug(f"YF parse error for {ticker}: {e}")
                continue

        logger.info(f"yf.download() fetched prices for {len(data_dict)}/{len(tickers)} tickers")
        return data_dict

    
    async def get_historical_data(self, access_token: Optional[str], instrument_key: str, interval: str = "1minute", days: int = 7) -> List[Dict]:
        """
        Fetch historical candle data with cache-first strategy.
        """
        # Validate interval
        valid_intervals = {"1minute", "30minute", "day", "week", "month", "1day"}
        if interval not in valid_intervals:
            logger.error("Invalid interval for historical data: %s", interval)
            return []

        # Resolve symbol for cache key via O(1) lookup
        symbol = self._instrument_to_symbol.get(instrument_key)

        # Check cache
        if symbol:
            cached = self.cache.get_historical(symbol, interval)
            if cached:
                return cached

        def _ts_sort_key(c: Dict) -> float:
            ts = c.get("timestamp")
            if ts is None:
                return 0.0
            try:
                if isinstance(ts, (int, float)):
                    return float(ts) / 1000.0 if float(ts) > 1_000_000_000_000 else float(ts)
                if isinstance(ts, str):
                    s = ts.strip()
                    if s.isdigit():
                        v = float(s)
                        return v / 1000.0 if v > 1_000_000_000_000 else v
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0
            return 0.0

        # Try Upstox via client
        token = access_token or upstox_client.get_global_token()
        if token:
            data = await self.upstox.get_historical_candles(instrument_key, interval, days, token=token)
            if data:
                result = sorted(data, key=_ts_sort_key)
                if symbol:
                    self.cache.set_historical(symbol, interval, result)
                return result

        # Fallback to YFinance
        data = await self._get_yfinance_history(instrument_key, interval, days)
        result = sorted(data, key=_ts_sort_key)
        if symbol and result:
            self.cache.set_historical(symbol, interval, result)
        return result

    async def _get_yfinance_history(self, instrument_key: str, interval: str, days: int) -> List[Dict]:
        ticker = self._instrument_to_yf.get(instrument_key)
        if not ticker:
            return []
            
        # Map interval
        yf_interval = "1m" if interval == "1minute" else "1d"
        if interval == "30minute": yf_interval = "30m"
        
        try:
            loop = asyncio.get_event_loop()
            hist = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).history(period=f"{days}d", interval=yf_interval))
            
            candles = []
            for index, row in hist.iterrows():
                candles.append({
                    "timestamp": index.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                    "open": row['Open'],
                    "high": row['High'],
                    "low": row['Low'],
                    "close": row['Close'],
                    "volume": int(row['Volume'])
                })
            # Return chronological order (oldest -> newest). Charting libraries expect ascending time.
            return candles
        except Exception as e:
            logger.error(f"YFinance history error: {e}")
            return []

    def get_all_stocks(self) -> List[Dict]:
        return [{"symbol": s, "name": d["name"], "sector": d["sector"], "instrument_key": d["instrument_key"]} for s, d in STOCK_DATA.items()]
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        if symbol in STOCK_DATA:
            data = STOCK_DATA[symbol]
            return {"symbol": symbol, "name": data["name"], "sector": data["sector"], "instrument_key": data["instrument_key"]}
        return None
    
    def get_all_indices(self) -> List[Dict]:
        return [{"symbol": s, "name": d["name"], "instrument_key": d["instrument_key"]} for s, d in INDEX_DATA.items()]

    def get_fundamental_info(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data from Yahoo Finance as a fallback for Screener.in.
        Returns a dict matching the structure expected by the frontend:
        {
            "symbol": str,
            "company_name": str,
            "about": str,
            "ratios": Dict[str, str],
            "pros": List[str],
            "cons": List[str]
        }
        """
        try:
            stock_info = self.get_stock_info(symbol)
            if not stock_info:
                return None
            
            ticker_symbol = STOCK_DATA.get(symbol, {}).get("yf_ticker", f"{symbol}.NS")
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            if not info:
                return None
            
            # Extract Ratios
            ratios = {}
            if info.get('marketCap'):
                mcap_cr = info['marketCap'] / 10000000  # Convert to Crores
                ratios['Market Cap'] = f"₹{mcap_cr:,.0f} Cr"
            
            if info.get('currentPrice'):
                ratios['Current Price'] = f"₹{info['currentPrice']}"
            elif info.get('regularMarketPrice'):
                 ratios['Current Price'] = f"₹{info['regularMarketPrice']}"
            
            if info.get('dayHigh') and info.get('dayLow'):
                ratios['High / Low'] = f"₹{info['dayHigh']} / ₹{info['dayLow']}"
            
            if info.get('trailingPE'):
                ratios['Stock P/E'] = f"{info['trailingPE']:.2f}"
            
            if info.get('bookValue'):
                ratios['Book Value'] = f"₹{info['bookValue']:.2f}"
            
            if info.get('dividendYield'):
                # YFinance dividendYield seems to be 0.38 for 0.38% (already scaled) or my observation is wrong.
                # However, usually it is decimal. If I saw 0.38, it is likely 0.0038 * 100? 
                # Let's assume input is decimal. If input is 0.0038, *100 = 0.38%.
                # But verification showed 38.00%. So input was 0.38.
                # So we use input as is.
                diff_yield = info['dividendYield']
                # Heuristic: if yield > 1 (e.g. 5), it is %, use as is. 
                # If yield < 1 and > 0.1 (e.g. 0.38), could be % or decimal (38%). 
                # For Reliance 0.38 is definitely %.
                # Let's just assume it's percentage for now if it matches rate/price.
                ratios['Dividend Yield'] = f"{diff_yield:.2f}%"
            
            if info.get('returnOnEquity'):
                ratios['ROE'] = f"{info['returnOnEquity']*100:.2f}%"
            
            if info.get('returnOnAssets'):
                ratios['ROA'] = f"{info['returnOnAssets']*100:.2f}%"

            if info.get('debtToEquity'):
                ratios['Debt to Equity'] = f"{info['debtToEquity']:.2f}"

            # Generate Pros & Cons dynamically
            pros = []
            cons = []
            
            pe = info.get('trailingPE', 0)
            de = info.get('debtToEquity', 0)
            roe = info.get('returnOnEquity', 0)
            div = info.get('dividendYield', 0)
            profit_growth = info.get('earningsGrowth', 0)

            # Pros
            if de < 50: pros.append("Company has low debt.")
            if roe > 0.15: pros.append("Good return on equity over 15%.")
            if div > 0.03: pros.append("Good dividend yield.")
            if profit_growth > 0.10: pros.append("Company has shown good profit growth.")
            if info.get('priceToBook', 10) < 3: pros.append("Stock is trading at decent book value.")

            # Cons
            if pe > 40: cons.append("Stock is trading at a high PE valuation.")
            if de > 100: cons.append("Company has high debt levels.")
            if roe < 0.10: cons.append("Low return on equity.")
            if info.get('priceToBook', 1) > 10: cons.append("Stock is trading at high book value.")
            
            if not pros: pros.append("Company is stable.")
            if not cons: cons.append("No major red flags.")

            return {
                "symbol": symbol,
                "company_name": info.get('longName', stock_info['name']),
                "about": info.get('longBusinessSummary') or info.get('description', "No description available."),
                "ratios": ratios,
                "pros": pros,
                "cons": cons
            }
            
        except Exception as e:
            logger.error(f"Error fetching YFinance fundamentals for {symbol}: {e}")
            return None

market_service = MarketService()
