import os
from pathlib import Path
import sqlite3

# Define the project structure
project_structure = {
    "stock_analysis_app": {
        "analysis": {
            "mxwll_suite_indicator.py": '''import pandas as pd
import plotly.graph_objects as go
from ta.volatility import AverageTrueRange
import warnings
from datetime import datetime, timedelta

def mxwll_suite_indicator(df, ticker, params):
    """
    Performs MXWLL Suite Indicator analysis on the provided DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing stock data.
        ticker (str): The stock ticker symbol.
        params (dict): Dictionary of parameters for the analysis.
    
    Returns:
        plotly.graph_objects.Figure: The resulting Plotly figure.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Extract data frequency from params
    data_frequency = params.get('data_frequency', '1D')
    
    # Derived parameters
    if data_frequency == '15m':
        atr_window = 14
        aoi_length = 50
        session_enabled = True
        session_times = {
            'New York': {'start': '09:30', 'end': '16:00'},
            'Asia': {'start': '20:00', 'end': '02:00'},
            'London': {'start': '03:00', 'end': '11:30'}
        }
    elif data_frequency == '4h':
        atr_window = 14
        aoi_length = 50
        session_enabled = True
        session_times = {
            'New York': {'start': '09:30', 'end': '16:00'},
            'Asia': {'start': '20:00', 'end': '02:00'},
            'London': {'start': '03:00', 'end': '11:30'}
        }
    elif data_frequency == '1D':
        atr_window = 14
        aoi_length = 50
        session_enabled = False
        session_times = {}
    else:
        raise ValueError("Invalid data_frequency. Choose from '15m', '4h', or '1D'.")
    
    # Session Colors
    session_colors = {
        'New York': params.get('bull_color', '#14D990'),
        'Asia': params.get('bear_color', '#F24968'),
        'London': params.get('fvg_color', '#F2B807')
    }
    
    # --- Helper Functions ---
    
    def calculate_pivots_vectorized(df, sensitivity):
        """
        Vectorized pivot calculation for enhanced performance.
        Identifies swing highs and lows based on the specified sensitivity.
        """
        print(f"Calculating pivots with sensitivity {sensitivity} using vectorized operations for {data_frequency} data...")
        # Swing Highs
        rolling_max = df['High'].rolling(window=2*sensitivity+1, center=True).max()
        swing_highs = df[df['High'] == rolling_max].index.to_list()
        
        # Swing Lows
        rolling_min = df['Low'].rolling(window=2*sensitivity+1, center=True).min()
        swing_lows = df[df['Low'] == rolling_min].index.to_list()
        
        print(f"Pivots calculated: {len(swing_highs)} swing highs and {len(swing_lows)} swing lows.")
        return swing_highs, swing_lows
    
    def identify_fvg(df):
        """
        Identifies Fair Value Gaps (FVG) in the data.
        """
        print("Identifying Fair Value Gaps (FVG)...")
        fvg_up = []
        fvg_down = []
        for i in range(1, len(df)):
            prev_high = df['High'].iloc[i-1]
            prev_low = df['Low'].iloc[i-1]
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]
            if prev_high < current_low:
                fvg_up.append({'x0': df.index[i], 'y0': prev_high, 'x1': df.index[i], 'y1': current_low})
            if prev_low > current_high:
                fvg_down.append({'x0': df.index[i], 'y0': current_high, 'x1': df.index[i], 'y1': prev_low})
        print(f"Fair Value Gaps identified: {len(fvg_up)} up FVGs and {len(fvg_down)} down FVGs.")
        return fvg_up, fvg_down
    
    def plot_fibonacci_levels(fig, last_high, last_low):
        """
        Plots Fibonacci retracement levels based on the latest swing high and low.
        """
        fib_levels = params.get('fib_levels', [0.236, 0.382, 0.5, 0.618, 0.786])
        fib_colors = params.get('fib_colors', ['gray', 'lime', 'yellow', 'orange', 'red'])
        show_fib5 = params.get('show_fib5', True)
        
        fib_diff = last_high - last_low
        print("Plotting Fibonacci levels...")
        for level, color in zip(fib_levels, fib_colors):
            if level == 0.5 and not show_fib5:
                continue
            fib_level = last_low + fib_diff * level
            fig.add_hline(y=fib_level, line=dict(color=color, dash='dash'), 
                          annotation_text=f'Fib {level}', annotation_position="top left")
    
    def draw_aoe(fig, df):
        """
        Draws the Area of Interest (AOE) boxes based on ATR and recent price action.
        """
        print("Drawing Area of Interest (AOE)...")
        atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=atr_window)
        df['ATR'] = atr_indicator.average_true_range()
        
        # Define AOE window
        aoi = df.iloc[-aoi_length:]
        max_aoi_high = max(aoi['High'].max(), aoi['Open'].iloc[-aoi_length:].max())
        min_aoi_low = min(aoi['Low'].min(), aoi['Open'].iloc[-aoi_length:].min())
        atr_latest = df['ATR'].iloc[-1]
        
        # High AOE Box
        high_aoe_y0 = max_aoi_high * 1.01
        high_aoe_y1 = max_aoi_high
        high_aoe_x0 = aoi.index[0]
        high_aoe_x1 = aoi.index[-1]
        
        fig.add_shape(type="rect",
                      x0=high_aoe_x0,
                      y0=high_aoe_y0,
                      x1=high_aoe_x1,
                      y1=high_aoe_y1,
                      fillcolor=params.get('bear_color', '#F24968'),
                      opacity=0.2,
                      line=dict(width=0),
                      layer='below',
                      name='High AOE')
        
        # Low AOE Box
        low_aoe_y0 = min_aoi_low
        low_aoe_y1 = min_aoi_low * 0.99
        low_aoe_x0 = aoi.index[0]
        low_aoe_x1 = aoi.index[-1]
        
        fig.add_shape(type="rect",
                      x0=low_aoe_x0,
                      y0=low_aoe_y0,
                      x1=low_aoe_x1,
                      y1=low_aoe_y1,
                      fillcolor=params.get('bull_color', '#14D990'),
                      opacity=0.2,
                      line=dict(width=0),
                      layer='below',
                      name='Low AOE')
        
        print("AOE drawn.")
    
    def highlight_sessions(fig, df):
        """
        Highlights trading sessions (New York, Asia, London) on the chart.
        Applicable only for intra-day data frequencies.
        """
        if not session_enabled:
            print("Session highlighting is disabled for this data frequency.")
            return
        
        print("Highlighting trading sessions...")
        for session, props in session_times.items():
            start_time = datetime.strptime(props['start'], '%H:%M').time()
            end_time = datetime.strptime(props['end'], '%H:%M').time()
            
            for date in df.index.normalize().unique():
                start_datetime = datetime.combine(date, start_time)
                end_datetime = datetime.combine(date, end_time)
                # Handle sessions that span over midnight
                if end_datetime <= start_datetime:
                    end_datetime += timedelta(days=1)
                
                fig.add_vrect(
                    x0=start_datetime,
                    x1=end_datetime,
                    fillcolor=session_colors.get(session, 'rgba(0,0,0,0)'),
                    opacity=params.get('transparency', 0.98),
                    layer="below",
                    line_width=0,
                    annotation_text=session if (date == df.index[-1].normalize()) else "",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="white",
                    name=session
                )
        print("Trading sessions highlighted.")
    
    def volume_activity(df):
        """
        Categorizes volume into different activity levels based on quantiles.
        """
        print("Analyzing volume activity...")
        vol_perc1 = df['Volume'].quantile(0.1)
        vol_perc2 = df['Volume'].quantile(0.33)
        vol_perc3 = df['Volume'].quantile(0.5)
        vol_perc4 = df['Volume'].quantile(0.66)
        vol_perc5 = df['Volume'].quantile(0.9)
        
        def categorize_volume(vol):
            if vol <= vol_perc1:
                return "Very Low"
            elif vol <= vol_perc2:
                return "Low"
            elif vol <= vol_perc3:
                return "Average"
            elif vol <= vol_perc4:
                return "High"
            else:
                return "Very High"
        
        df['VolumeActivity'] = df['Volume'].apply(categorize_volume)
        print("Volume activity categorized.")
        return df
    
    def draw_main_line(fig, big_upper, big_lower):
        """
        Draws the main line connecting the latest swing high and low.
        """
        if not big_upper or not big_lower:
            print("Insufficient data to draw the main line.")
            return None
        
        # Get the latest swing high and low
        latest_swing_high = big_upper[-1]
        latest_swing_low = big_lower[-1]
        
        # Define main line based on the latest swing points
        main_line = {
            'x1': latest_swing_low,
            'y1': df.loc[latest_swing_low, 'Low'],
            'x2': latest_swing_high,
            'y2': df.loc[latest_swing_high, 'High']
        }
        
        fig.add_trace(go.Scatter(
            x=[main_line['x1'], main_line['x2']],
            y=[main_line['y1'], main_line['y2']],
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Main Line'
        ))
        
        return main_line
    
    def draw_fibs(fig, main_line):
        """
        Draws Fibonacci retracement levels based on the main line.
        """
        if not main_line:
            return
        
        last_high = main_line['y2']
        last_low = main_line['y1']
        plot_fibonacci_levels(fig, last_high, last_low)
    
    def add_volume_annotation(fig, df):
        """
        Adds a volume activity annotation to the latest data point.
        """
        print("Adding volume activity annotation...")
        latest = df.iloc[-1]
        latest_time = latest.name
        latest_volume_activity = latest['VolumeActivity']
        
        current_session = "Dead Zone"
        time_until_change = "N/A"
        
        if data_frequency in ['15m', '4h']:
            # Determine current session based on the latest timestamp
            for session, props in session_times.items():
                start_time = datetime.strptime(props['start'], '%H:%M').time()
                end_time = datetime.strptime(props['end'], '%H:%M').time()
                if start_time <= latest_time.time() <= end_time:
                    current_session = session
                    break
            
            # Calculate time until next session change
            def calculate_time_until_change(current_time):
                print("Calculating time until next session change...")
                for session, props in session_times.items():
                    start_time = datetime.strptime(props['start'], '%H:%M').time()
                    end_time = datetime.strptime(props['end'], '%H:%M').time()
                    if start_time <= current_time.time() <= end_time:
                        # Next session is the following session in the list
                        session_names = list(session_times.keys())
                        current_index = session_names.index(session)
                        next_session = session_names[(current_index + 1) % len(session_names)]
                        next_start_str = session_times[next_session]['start']
                        next_start_time = datetime.strptime(next_start_str, '%H:%M').time()
                        next_start_datetime = datetime.combine(current_time.date(), next_start_time)
                        if next_start_time <= current_time.time():
                            next_start_datetime += timedelta(days=1)
                        time_diff = next_start_datetime - current_time
                        hours, remainder = divmod(int(time_diff.total_seconds()), 3600)
                        minutes, _ = divmod(remainder, 60)
                        print(f"Time until change: {hours}h {minutes}m")
                        return f"{hours}h {minutes}m"
                print("Time until change: N/A")
                return "N/A"
            
            time_until_change = calculate_time_until_change(latest_time)
        else:
            # For daily data, sessions are not time-based
            current_session = "Dead Zone"
        
        annotation_text = f"""
        Session: {current_session}<br>
        Session Close: {time_until_change}<br>
        Volume Activity: {latest_volume_activity}
        """
        
        fig.add_annotation(
            x=latest_time,
            y=latest['High'],
            text=annotation_text,
            showarrow=True,
            arrowhead=1,
            align="left",
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        )
        print("Volume activity annotation added.")
    
    # --- Calculate Pivots ---
    print("Starting pivot calculations...")
    big_upper, big_lower = calculate_pivots_vectorized(df, params.get('external_sensitivity', 25))
    if params.get('show_internals', True):
        small_upper, small_lower = calculate_pivots_vectorized(df, params.get('internal_sensitivity', 3))
    else:
        small_upper, small_lower = [], []
    
    # --- Identify FVG ---
    fvg_up, fvg_down = identify_fvg(df)
    
    # --- Volume Activity ---
    df = volume_activity(df)
    
    # --- Create Plotly Figure ---
    print("Creating Plotly figure...")
    fig = go.Figure()
    
    # --- Plot Candlestick ---
    print("Adding candlestick chart...")
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # --- Plot Swing Highs ---
    print("Plotting big swing highs...")
    for swing_time in big_upper[-params.get('swing_order_blocks', 10):]:
        try:
            swing_price = df.loc[swing_time, 'High']
        except KeyError:
            print(f"Warning: No 'High' price found for swing_time {swing_time}. Skipping this swing high.")
            continue
        label_text = 'HH'
        if params.get('show_hhlh', True):
            fig.add_trace(go.Scatter(
                x=[swing_time],
                y=[swing_price],
                mode='markers+text',
                marker=dict(color=params.get('bear_color', '#F24968'), size=10, symbol='triangle-up'),
                text=[label_text],
                textposition='bottom center',
                name='Swing High'
            ))
    
    # --- Plot Swing Lows ---
    print("Plotting big swing lows...")
    for swing_time in big_lower[-params.get('swing_order_blocks', 10):]:
        try:
            swing_price = df.loc[swing_time, 'Low']
        except KeyError:
            print(f"Warning: No 'Low' price found for swing_time {swing_time}. Skipping this swing low.")
            continue
        label_text = 'LL'
        if params.get('show_hlll', True):
            fig.add_trace(go.Scatter(
                x=[swing_time],
                y=[swing_price],
                mode='markers+text',
                marker=dict(color=params.get('bull_color', '#14D990'), size=10, symbol='triangle-down'),
                text=[label_text],
                textposition='top center',
                name='Swing Low'
            ))
    
    # --- Plot Internal Swing Highs ---
    if params.get('show_internals', True):
        print("Plotting internal swing highs...")
        for swing_time in small_upper:
            try:
                swing_price = df.loc[swing_time, 'High']
            except KeyError:
                print(f"Warning: No 'High' price found for swing_time {swing_time}. Skipping this internal swing high.")
                continue
            fig.add_trace(go.Scatter(
                x=[swing_time],
                y=[swing_price],
                mode='markers',
                marker=dict(color=params.get('bear_color', '#F24968'), size=6, symbol='triangle-up'),
                name='Internal Swing High'
            ))
    
    # --- Plot Internal Swing Lows ---
    if params.get('show_internals', True):
        print("Plotting internal swing lows...")
        for swing_time in small_lower:
            try:
                swing_price = df.loc[swing_time, 'Low']
            except KeyError:
                print(f"Warning: No 'Low' price found for swing_time {swing_time}. Skipping this internal swing low.")
                continue
            fig.add_trace(go.Scatter(
                x=[swing_time],
                y=[swing_price],
                mode='markers',
                marker=dict(color=params.get('bull_color', '#14D990'), size=6, symbol='triangle-down'),
                name='Internal Swing Low'
            ))
    
    # --- Plot Fair Value Gaps (FVG) ---
    if params.get('show_fvg', True):
        print("Plotting Fair Value Gaps (FVG)...")
        for gap in fvg_up:
            fig.add_shape(type="rect",
                          x0=gap['x0'],
                          y0=gap['y0'],
                          x1=gap['x1'],
                          y1=gap['y1'],
                          fillcolor=params.get('fvg_color', '#F2B807'),
                          opacity=params.get('fvg_transparency', 80) / 100,
                          line=dict(width=0),
                          layer='below',
                          name='FVG Up')
        for gap in fvg_down:
            fig.add_shape(type="rect",
                          x0=gap['x0'],
                          y0=gap['y0'],
                          x1=gap['x1'],
                          y1=gap['y1'],
                          fillcolor=params.get('fvg_color', '#F2B807'),
                          opacity=params.get('fvg_transparency', 80) / 100,
                          line=dict(width=0),
                          layer='below',
                          name='FVG Down')
        print("FVG plotted.")
    
    # --- Draw Area of Interest (AOE) ---
    if params.get('show_aoe', True):
        draw_aoe(fig, df)
    
    # --- Highlight Trading Sessions ---
    highlight_sessions(fig, df)
    
    # --- Draw Main Line (Connecting Latest Swing Points) ---
    main_line = draw_main_line(fig, big_upper, big_lower)
    
    # --- Draw Fibonacci Levels ---
    if main_line and params.get('show_fibs', True):
        draw_fibs(fig, main_line)
    
    # --- Add Volume Activity Annotation ---
    add_volume_annotation(fig, df)
    
    # --- Final Layout Adjustments ---
    print("Finalizing layout...")
    fig.update_layout(
        title=f'Mxwll Suite Indicator for {ticker}',
        yaxis_title='Price',
        xaxis_title='Date',
        legend=dict(orientation="h"),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    print("Layout finalized.")
    
    return fig
    ''',
        },
        "data": {
            "tick_data.db": ""  # Empty SQLite database will be created automatically
        },
        "utils": {
            "db_manager.py": '''import sqlite3
import os
from datetime import datetime, timedelta

def initialize_db(db_path='data/tick_data.db'):
    """
    Initializes the SQLite database and creates the 'Ticker' table if it doesn't exist.
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
        ChangeP REAL,
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
        return conn
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        return None

def get_tickers_from_db(conn):
    """
    Retrieves a list of unique tickers from the database.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT Ticker FROM Ticker;")
        fetched_tickers = cursor.fetchall()
        tickers = [row[0] for row in fetched_tickers]
        return tickers
    except sqlite3.Error as e:
        print(f"Failed to retrieve tickers: {e}")
        return []

def get_latest_date_for_ticker(conn, ticker):
    """
    Retrieves the latest date for a given ticker from the database.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM Ticker WHERE Ticker = ?;", (ticker,))
        result = cursor.fetchone()
        return result[0] if result and result[0] else None
    except sqlite3.Error as e:
        print(f"Failed to retrieve latest date for ticker '{ticker}': {e}")
        return None

def insert_data_into_db(conn, df, ticker, batch_size=100):
    """
    Inserts the DataFrame data into the SQLite database in batches.
    """
    try:
        cursor = conn.cursor()
        records = df.to_records(index=False)
        data_to_insert = [(ticker, *record) for record in records]
        
        insert_query = """
        INSERT OR IGNORE INTO Ticker 
        (Ticker, Date, Open, High, Low, Close, Change, ChangeP, Volume) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            conn.commit()
            print(f"Inserted batch {i // batch_size + 1} with {len(batch)} records for ticker '{ticker}'.")
        print(f"All data for ticker '{ticker}' successfully inserted into the database.")
    except sqlite3.Error as e:
        print(f"Failed to insert data into database for ticker '{ticker}': {e}")

def add_new_ticker(conn, ticker, data_fetcher_func):
    """
    Adds a new ticker by fetching its data and inserting into the database.
    """
    date_from = "01 Jan 2020"
    date_to = datetime.today().strftime("%d %b %Y")
    
    raw_data = data_fetcher_func(ticker, date_from, date_to)
    if raw_data:
        df = process_data(raw_data)
        if df is not None and not df.empty:
            insert_data_into_db(conn, df, ticker)
            return True
    return False

def process_data(raw_data):
    """
    Processes raw stock data into a pandas DataFrame.
    """
    import pandas as pd
    from dateutil import parser
    
    df = pd.DataFrame(raw_data)
    
    required_columns = ["Date_", "Open", "High", "Low", "Close", "Change", "ChangeP", "Volume"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing expected column in data: {col}")
            return None
    
    # Format the date using dateutil.parser for flexibility
    df['Date'] = df['Date_'].apply(lambda x: format_date(x))
    if df['Date'].isnull().any():
        print("One or more dates failed to parse.")
        return None
    
    # Select and reorder columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'ChangeP', 'Volume']]
    
    # Rename columns for clarity
    df.rename(columns={
        'ChangeP': 'Change (%)'
    }, inplace=True)
    
    # Round numerical columns to two decimal places without converting to strings
    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Change', 'Change (%)']
    df[numerical_cols] = df[numerical_cols].round(2)
    
    # Ensure 'Volume' is integer
    try:
        df['Volume'] = df['Volume'].astype(int)
    except ValueError as e:
        print(f"Error converting Volume to integer: {e}")
        return None
    
    return df

def format_date(date_str, output_format="%Y-%m-%d"):
    """
    Parses the date string and formats it to the desired output format.
    """
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime(output_format)
    except (ValueError, OverflowError) as e:
        print(f"Error: Unable to parse date '{date_str}': {e}")
        return None
    ''',
            "data_fetcher.py": '''import requests
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
        "Sec-CH-UA": '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
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
    ''',
        },
        "main.py": '''import streamlit as st
import pandas as pd
from utils.db_manager import (
    initialize_db, 
    get_tickers_from_db, 
    get_latest_date_for_ticker, 
    insert_data_into_db, 
    add_new_ticker, 
    process_data
)
from utils.data_fetcher import get_stock_data
from analysis.mxwll_suite_indicator import mxwll_suite_indicator

import os

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Data Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the App
st.title("📈 Stock Data Analyzer")

# Initialize database
conn = initialize_db()

if conn is None:
    st.error("Failed to connect to the database. Please check the logs.")
    st.stop()

# Sidebar for navigation
st.sidebar.header("Options")

app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Synchronize Database", "Add New Ticker", "Analyze Tickers"])

# Function to synchronize database
def synchronize_database():
    st.header("🔄 Synchronize Database")
    if st.button("Start Synchronization"):
        tickers = get_tickers_from_db(conn)
        if not tickers:
            st.warning("No tickers found in the database. Please add new tickers first.")
            return
        
        status_text = st.empty()
        for idx, ticker in enumerate(tickers, start=1):
            status_text.text(f"Processing ticker {idx}/{len(tickers)}: {ticker}")
            
            latest_date_str = get_latest_date_for_ticker(conn, ticker)
            if latest_date_str:
                latest_date = pd.to_datetime(latest_date_str)
                date_from_dt = latest_date + pd.Timedelta(days=1)
                date_from = date_from_dt.strftime("%d %b %Y")
                if date_from_dt > pd.Timestamp.today():
                    status_text.text(f"No new data to fetch for ticker '{ticker}'. Already up to date.")
                    continue
            else:
                date_from = "01 Jan 2020"
            
            date_to = pd.Timestamp.today().strftime("%d %b %Y")
            
            raw_data = get_stock_data(ticker, date_from, date_to)
            if raw_data:
                df = process_data(raw_data)
                if df is not None and not df.empty:
                    insert_data_into_db(conn, df, ticker)
                    st.success(f"Data for ticker '{ticker}' updated successfully.")
                else:
                    st.warning(f"No valid data to process for ticker '{ticker}'.")
            else:
                st.error(f"Failed to retrieve data for ticker '{ticker}'.")
        
        status_text.text("Synchronization complete.")

# Function to add new tickers
def add_new_ticker_ui():
    st.header("➕ Add New Ticker")
    ticker_input = st.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT):").upper()
    if st.button("Add Ticker"):
        if ticker_input:
            tickers_in_db = get_tickers_from_db(conn)
            if ticker_input in tickers_in_db:
                st.warning(f"Ticker '{ticker_input}' already exists in the database.")
            else:
                raw_data = get_stock_data(ticker_input, "01 Jan 2020", pd.Timestamp.today().strftime("%d %b %Y"))
                if raw_data:
                    df = process_data(raw_data)
                    if df is not None and not df.empty:
                        insert_data_into_db(conn, df, ticker_input)
                        st.success(f"Ticker '{ticker_input}' added successfully.")
                    else:
                        st.error(f"No valid data to add for ticker '{ticker_input}'.")
                else:
                    st.error(f"Failed to retrieve data for ticker '{ticker_input}'.")
        else:
            st.error("Please enter a valid ticker symbol.")

# Function to analyze tickers
def analyze_tickers():
    st.header("🔍 Analyze Tickers")
    tickers = get_tickers_from_db(conn)
    if not tickers:
        st.warning("No tickers available for analysis. Please add tickers first.")
        return
    
    selected_tickers = st.multiselect("Select Tickers for Analysis", tickers)
    
    if selected_tickers:
        for ticker in selected_tickers:
            st.subheader(f"📊 Analysis for {ticker}")
            
            # Fetch data from database
            query = f"SELECT * FROM Ticker WHERE Ticker = ? ORDER BY Date ASC;"
            df = pd.read_sql_query(query, conn, params=(ticker,))
            if df.empty:
                st.warning(f"No data available for ticker '{ticker}'.")
                continue
            
            # Convert 'Date' column to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Define analysis parameters
            params = {
                "bull_color": '#14D990',
                "bear_color": '#F24968',
                "show_internals": True,
                "internal_sensitivity": 3,  # Options: 3, 5, 8
                "internal_structure": "All",  # Options: "All", "BoS", "CHoCH"
                "show_externals": True,
                "external_sensitivity": 25,  # Options: 10, 25, 50
                "external_structure": "All",  # Options: "All", "BoS", "CHoCH"
                "show_order_blocks": True,
                "swing_order_blocks": 10,
                "show_hhlh": True,
                "show_hlll": True,
                "show_aoe": True,
                "show_prev_day_high": True,
                "show_prev_day_labels": True,
                "show_4h_high": True,
                "show_4h_labels": True,
                "show_fvg": True,
                "contract_violated_fvg": False,
                "close_only_fvg": False,
                "fvg_color": '#F2B807',
                "fvg_transparency": 80,  # Percentage
                "show_fibs": True,
                "show_fib236": True,
                "show_fib382": True,
                "show_fib5": True,
                "show_fib618": True,
                "show_fib786": True,
                "fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
                "fib_colors": ['gray', 'lime', 'yellow', 'orange', 'red'],
                "transparency": 0.98,  # For session highlighting
                "data_frequency": '1D'  # Adjust as needed
            }
            
            # Perform analysis
            try:
                fig = mxwll_suite_indicator(df, ticker, params)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during analysis for ticker '{ticker}': {e}")

# Render different app modes
if app_mode == "Synchronize Database":
    synchronize_database()
elif app_mode == "Add New Ticker":
    add_new_ticker_ui()
elif app_mode == "Analyze Tickers":
    analyze_tickers()
''',
        },
        "requirements.txt": '''streamlit
pandas
plotly
ta
requests
tabulate
python-dateutil
''',
    }


def create_project_structure(base_path=Path("."), structure=project_structure):
    """
    Recursively creates directories and files based on the provided structure.

    Args:
        base_path (Path): The base directory where the structure will be created.
        structure (dict): A nested dictionary representing the directory and file structure.
    """
    for name, content in structure.items():
        current_path = base_path / name
        if isinstance(content, dict):
            # Create directory
            current_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {current_path}")
            # Recursively create subdirectories/files
            create_project_structure(current_path, content)
        else:
            # Create and write to file
            file_path = current_path
            if file_path.exists():
                print(f"File already exists and will be skipped: {file_path}")
            else:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(content)
                print(f"Created file: {file_path}")

def create_empty_db(db_path='stock_analysis_app/data/tick_data.db'):
    """
    Creates an empty SQLite database if it doesn't exist.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        conn.close()
        print(f"Created empty SQLite database at: {db_path}")
    else:
        print(f"SQLite database already exists at: {db_path}")

if __name__ == "__main__":
    # Create project structure
    create_project_structure()
    
    # Create empty SQLite database
    create_empty_db()
    
    print("\nProject structure created successfully!")
    print("You can now navigate to 'stock_analysis_app/' and start using your Streamlit app.")
