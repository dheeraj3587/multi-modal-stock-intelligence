import requests
from bs4 import BeautifulSoup
import logging

class ScreenerService:
    def __init__(self):
        self.base_url = "https://www.screener.in/company"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        self.logger = logging.getLogger(__name__)

    def get_company_details(self, symbol, company_name=None):
        """
        Scrapes company details from Screener.in.
        First tries to find the company URL via search.
        """
        try:
            # 1. Initial Search
            # Try searching by symbol first (cleaned)
            clean_symbol = symbol.split('.')[0]
            search_query = clean_symbol
            
            company_url = self._search_company(search_query)
            
            # If prompt failed and we have a company name, try that
            if not company_url and company_name:
                self.logger.info(f"Retrying search with company name: {company_name}")
                company_url = self._search_company(company_name)
            
            # If still found nothing, and symbol looks like a standard ticker, try direct URL as last resort
            if not company_url:
                company_url = f"{self.base_url}/{clean_symbol}/"
            
            # 2. Fetch Data
            self.logger.info(f"Fetching data from Screener: {company_url}")
            response = requests.get(company_url, headers=self.headers)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch data from Screener. Status: {response.status_code}")
                # If we tried strict search and failed, and haven't tried direct URL yet, maybe try that? 
                # But we constructed company_url either from search or direct. So if it fails, it fails.
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {
                "symbol": symbol,
                "company_name": "",
                "about": "",
                "ratios": {},
                "pros": [],
                "cons": []
            }

            # 1. Get Company Name
            h1 = soup.find('h1')
            if h1:
                data["company_name"] = h1.text.strip()

            # 2. Get About Text
            about_section = soup.find('div', {'class': 'company-profile-about'})
            if about_section:
                p_tag = about_section.find('p')
                if p_tag:
                    data["about"] = p_tag.text.strip()
            
            # Alternative About selector
            if not data["about"]:
                about_div = soup.find('div', {'id': 'about'})
                if about_div:
                    p = about_div.find('p')
                    if p:
                        data["about"] = p.text.strip()

            # 3. Get Key Ratios (Top Section)
            ratios_div = soup.find('div', {'class': 'company-ratios'})
            if ratios_div:
                ul = ratios_div.find('ul', {'id': 'top-ratios'})
                if ul:
                    for li in ul.find_all('li'):
                        name = li.find('span', {'class': 'name'})
                        value = li.find('span', {'class': 'number'})
                        if name and value:
                            key = name.text.strip()
                            val = value.text.strip()
                            data["ratios"][key] = val

            # 4. Get Pros and Cons
            pros_section = soup.find('div', {'class': 'pros'})
            if pros_section:
                ul = pros_section.find('ul')
                if ul:
                    data["pros"] = [li.text.strip() for li in ul.find_all('li')]

            cons_section = soup.find('div', {'class': 'cons'})
            if cons_section:
                ul = cons_section.find('ul')
                if ul:
                    data["cons"] = [li.text.strip() for li in ul.find_all('li')]
            
            return data

        except Exception as e:
            self.logger.error(f"Error scraping Screener.in: {str(e)}")
            return None

    def _search_company(self, query):
        """
        Search for company on Screener and return the URL slug
        """
        try:
            url = f"https://www.screener.in/api/company/search/?q={query}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                results = resp.json()
                if results:
                    # Prefer exact match if possible? 
                    # For now just take the first one
                    return f"https://www.screener.in{results[0]['url']}"
            return None
        except Exception as e:
            self.logger.error(f"Error searching Screener.in: {str(e)}")
            return None
