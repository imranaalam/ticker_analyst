import requests
import json

def get_stock_data(ticker, date_from, date_to):
    """
    Fetches stock data from the Investors Lounge API for a given ticker and date range.
    
    Args:
        ticker (str): The stock ticker symbol.
        date_from (str): Start date in 'DD MMM YYYY' format.
        date_to (str): End date in 'DD MMM YYYY' format.
    
    Returns:
        list: List of stock data dictionaries or None if failed.
    """
    url = "https://www.investorslounge.com/Default/SendPostRequest"
    
    headers = {
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9,ps;q=0.8",
        "Content-Type": "application/json; charset=UTF-8",
        "Priority": "u=1, i",
        "Sec-CH-UA": "\"Google Chrome\";v=\"129\", \"Not=A?Brand\";v=\"8\", \"Chromium\";v=\"129\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    payload = {
        "url": "PriceHistory/GetPriceHistoryCompanyWise",
        "data": json.dumps({
            "company": ticker,
            "sort": "0",
            "DateFrom": date_from,
            "DateTo": date_to,
            "key": ""
        })
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed for ticker '{ticker}': {e}")
        return None
    
    try:
        data = response.json()
        if not isinstance(data, list):
            print(f"Unexpected JSON structure for ticker '{ticker}': Expected a list of records.")
            return None
        print(f"[DEBUG] Retrieved {len(data)} records for ticker '{ticker}'.")
        return data
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response for ticker '{ticker}'.")
        return None
