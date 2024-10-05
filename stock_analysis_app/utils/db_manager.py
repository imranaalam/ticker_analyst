# File: stock_analysis_app/utils/db_manager.py

import sqlite3
from datetime import datetime
from dateutil import parser
import re  # Import regex module for processing Volume

def initialize_db(db_path='data/tick_data.db'):
    """
    Initializes the SQLite database and creates the 'Ticker' table with 'Volume' as INTEGER.
    
    Args:
        db_path (str): Path to the SQLite database file.
    
    Returns:
        sqlite3.Connection: SQLite connection object.
    """
    table_creation_query = """
    CREATE TABLE IF NOT EXISTS Ticker (
        Ticker TEXT NOT NULL,
        Date TEXT NOT NULL,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Change REAL,
        "Change (%)" REAL,
        Volume INTEGER,
        PRIMARY KEY (Ticker, Date)
    );
    """
    
    create_index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_ticker ON Ticker (Ticker);",
        "CREATE INDEX IF NOT EXISTS idx_date ON Ticker (Date);"
    ]
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(table_creation_query)
        for query in create_index_queries:
            cursor.execute(query)
        conn.commit()
        print(f"Database initialized at {db_path}.")
        return conn
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        return None

def get_tickers_from_db(conn):
    """
    Retrieves a list of unique tickers from the database.
    
    Args:
        conn (sqlite3.Connection): SQLite connection object.
    
    Returns:
        list: List of unique ticker symbols.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT Ticker FROM Ticker;")
        fetched_tickers = cursor.fetchall()
        tickers = [row[0] for row in fetched_tickers]
        print(f"Retrieved tickers from DB: {tickers}")
        return tickers
    except sqlite3.Error as e:
        print(f"Failed to retrieve tickers from the database: {e}")
        return []

def get_latest_date_for_ticker(conn, ticker):
    """
    Retrieves the latest date for a given ticker from the database.
    
    Args:
        conn (sqlite3.Connection): SQLite connection object.
        ticker (str): The stock ticker symbol.
    
    Returns:
        str: Latest date in 'YYYY-MM-DD' format or None if no data exists.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM Ticker WHERE Ticker = ?;", (ticker,))
        result = cursor.fetchone()
        latest_date = result[0] if result and result[0] else None
        print(f"Latest date for ticker '{ticker}': {latest_date}")
        return latest_date
    except sqlite3.Error as e:
        print(f"Failed to retrieve latest date for ticker '{ticker}': {e}")
        return None

def insert_data_into_db(conn, data, ticker, batch_size=100):
    """
    Inserts the list of stock data into the SQLite database in batches.
    
    Args:
        conn (sqlite3.Connection): SQLite connection object.
        data (list of dict): List of stock data dictionaries.
        ticker (str): The stock ticker symbol.
        batch_size (int): Number of records to insert per batch.
    
    Returns:
        bool: True if insertion is successful, False otherwise.
    """
    try:
        cursor = conn.cursor()
        insert_query = """
        INSERT OR IGNORE INTO Ticker 
        (Ticker, Date, Open, High, Low, Close, Change, "Change (%)", Volume) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        # Prepare data for insertion
        data_to_insert = []
        for record in data:
            # Extract and validate each field
            date = format_date(record.get("Date_"))
            if not date:
                print(f"Invalid date format in record: {record}")
                continue  # Skip records with invalid dates
            
            try:
                # Round numerical fields to two decimal places
                open_ = round(float(record.get("Open", 0)), 2)
                high = round(float(record.get("High", 0)), 2)
                low = round(float(record.get("Low", 0)), 2)
                close = round(float(record.get("Close", 0)), 2)
                change = round(float(record.get("Change", 0)), 2)
                change_p = round(float(record.get("ChangeP", 0)), 2)
                
                # Handle Volume: ensure it's an integer
                volume_raw = record.get("Volume", "0")
                if isinstance(volume_raw, (int, float)):
                    volume = int(volume_raw)
                elif isinstance(volume_raw, str):
                    # Check if Volume contains multiple comma-separated values
                    if ',' in volume_raw or '.' in volume_raw:
                        # Split by commas and periods
                        volume_parts = re.split(r'[.,]', volume_raw)
                        # Sum all numeric parts to derive a single Volume value
                        try:
                            volume = sum(int(part) for part in volume_parts if part.isdigit())
                        except ValueError:
                            print(f"Error converting Volume parts '{volume_parts}' to int for record: {record}")
                            continue  # Skip records with invalid Volume
                    else:
                        # Remove any non-digit characters and convert to integer
                        volume_cleaned = re.sub(r'[^\d]', '', volume_raw)
                        try:
                            volume = int(volume_cleaned)
                        except ValueError:
                            print(f"Error converting Volume '{volume_cleaned}' to int for record: {record}")
                            continue  # Skip records with invalid Volume
                else:
                    print(f"Unexpected Volume type in record: {record}")
                    continue  # Skip records with unexpected Volume type
            except (ValueError, TypeError) as e:
                print(f"Error processing record {record}: {e}")
                continue  # Skip records with conversion errors
            
            data_to_insert.append((
                ticker,
                date,
                open_,
                high,
                low,
                close,
                change,
                change_p,
                volume
            ))
        
        if not data_to_insert:
            print(f"No valid data to insert for ticker '{ticker}'.")
            return False
        
        # Insert data in batches
        total_records = len(data_to_insert)
        for i in range(0, total_records, batch_size):
            batch = data_to_insert[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            conn.commit()
            print(f"Inserted batch {i // batch_size + 1} with {len(batch)} records for ticker '{ticker}'.")
        
        print(f"All data for ticker '{ticker}' successfully inserted into the database.")
        return True
    except sqlite3.Error as e:
        print(f"Failed to insert data into database for ticker '{ticker}': {e}")
        return False

def add_new_ticker(conn, ticker, data_fetcher_func):
    """
    Adds a new ticker by fetching its data and inserting into the database.
    
    Args:
        conn (sqlite3.Connection): SQLite connection object.
        ticker (str): The stock ticker symbol.
        data_fetcher_func (function): Function to fetch stock data.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    date_from = "01 Jan 2020"
    date_to = datetime.today().strftime("%d %b %Y")
    
    raw_data = data_fetcher_func(ticker, date_from, date_to)
    if raw_data:
        success = insert_data_into_db(conn, raw_data, ticker)
        if success:
            print(f"Ticker '{ticker}' added successfully.")
            return True
    print(f"Failed to add ticker '{ticker}'.")
    return False

def format_date(date_str, output_format="%Y-%m-%d"):
    """
    Parses the date string and formats it to the desired output format.
    
    Args:
        date_str (str): The date string to parse.
        output_format (str): The desired format of the date string.
    
    Returns:
        str: Formatted date string or None if parsing fails.
    """
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime(output_format)
    except (ValueError, OverflowError) as e:
        print(f"Error: Unable to parse date '{date_str}': {e}")
        return None
