### Stock Analysis Dashboard using Dash, Plotly, and Alpha Vantage API.

### This script creates an interactive dashboard that:
### 1. Fetches stock price data from the Alpha Vantage API
### 1.1 The INPUT is from a UI that uses Ticker Textbox (for stock codes) and Interval Dropdown (for choosing intervals)
### 1.2 The DATA is fetched from Alpha Vantage (we choose to use closing prices at each interval)
### We acknowledge that the Data sourced is from Alpha Vantage API (https://www.alphavantage.co)
### 2. The OUTPUT has TWO parts
### 2.1 Single Number KPI: Displays stock KPIs (max, min, mean, variance, last % change at last closing price)
### 2.2 Graph KPI: Plots historical closing prices with 7-interval and 30-interval moving averages for close analysis



import pandas as pd
import requests # To make HTTP requests to the Alpha Vantage API
import dash
from dash import dcc, html, Input, Output # Dash components for layout, inputs, and outputs
import plotly.graph_objects as go

# Start the Dash app

app = dash.Dash()

# Create the layout of the Dash App
# UI INPUT and OUTPUT components on browser page

app.layout = html.Div(
    children = [
        # Page Title and brief introductions and prompts
        html.H1('STOCK ANALYSIS'), # Main heading
        html.H2('Now please key in the stock symbol in the textboxes below.'),
        html.H2('You may enter the stock sticker to analyze. Example: IBM, AAPL, AMZN'),

# Stock input field - Ticker Input Textbook for entering the stock ticker

        html.H2('Stock of Interest:'),
        dcc.Input(
            id = 'stock-input',
            type = 'text',
            placeholder = 'e.g., IBM',
            value = 'IBM', # starting value is IBM for illustration purposes
            style = {'width': '300px'}
        ),
        html.Br(),

# Time interval selection - dropdown for Daily, Weekly or Monthly
        html.H2('Now please select the intervals: '),
        dcc.Dropdown(
            id = 'time-dropdown',
            options = [
                {'label': 'Daily', 'value': 'daily'},
                {'label': 'Weekly', 'value': 'weekly'},
                {'label': 'Monthly', 'value': 'monthly'},
            ],
            value = 'daily', #starting value is daily for illustration purposes
        style = {'width': '300px'}
        ),

# Placeholders for the KPI display - OUTPUT
        html.H2('Analysis in numbers as below: '),
        html.H2('During the last 100 intervals, for the said stock '),

# Single Number KPIs: max price, min price, most recent % change, mean, and variance over selected intervals.
        html.H2(['Max Price is ',html.Span(id='max_price')]),
        html.H2(['Min Price is ',html.Span(id='min_price')]),
        html.H2(['Most recent price change is ', html.Span(id='last_pct_change')]),
        html.H2(['Mean Price is ',html.Span(id='mean_price')]),
        html.H2(['Variance is ',html.Span(id='variance')]),

# Placeholder for the graph
# Graph KPI: Plots historical closing prices with 7-interval and 30-interval moving averages for close analysis
        html.H2('Analysis in Graph as below: '),
        dcc.Graph(id="stock-graph")
    ]
)

# Callback function: compute single number KPIs and update graph KPI
@app.callback(
    Output("max_price", "children"),
    Output("min_price", "children"),
    Output("last_pct_change", "children"),
    Output("mean_price", "children"),
    Output("variance", "children"),
    Output("stock-graph", "figure"),
    Input("stock-input", "value"),
    Input("time-dropdown", "value")
)
def update_graph(symbol, interval):
    """
    Dash callback that:
      1) Receives the current ticker (symbol) and interval inputs from the UI Input
      2) Retrieves price data via get_stock_data() using API call
      3) Computes KPIs via calculating_KPIs_single_stock() * another function see below,
      4) Constructs a Plotly figure with Close, MA(7), MA(30),
      5) Returns back the KPI package of multiple strings and the figure to update the UI OUTPUT.

    Parameters
    ----------
    symbol : string | None
        Stock ticker input from the text field (e.g., "IBM").
    interval : string | None
        Interval from the dropdown: "daily", "weekly", or "monthly".

    Returns
    -------
    tuple
        (max_price, min_price, change_pct, mean_price, variance, figure)
        where the first five are strings for display and the last is a Plotly Figure.
    """


    symbol = (symbol or '').strip().upper()         # Clean and format inputs
    interval = (interval or '').strip().lower()

    df = get_stock_data(symbol, interval)           # Data fetch
    fig = go.Figure()                               # prepare an empty figure for use

    if df is None or df.empty:                      # When no data is available return NA status
        fig.update_layout(title = 'No Data', xaxis_title = 'Date', yaxis_title = 'Price')
        return 'na','na','na','na','na',fig

    analysis_result = calculating_KPIs_single_stock(df)     # Perform KPI calculations

# Add price and moving average traces - the point price series, the moving average (7) series and the moving average (30) series
# All series are based on common x-axis as time and y-axis as price
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Closing Price"))
    fig.add_trace(go.Scatter(x=df.index, y=analysis_result['ma7_series'], mode="lines", name="7 Intervals MA"))
    fig.add_trace(go.Scatter(x=df.index, y=analysis_result['ma30_series'], mode="lines", name="30 Intervals MA"))

# The layout configuration
    fig.update_layout(
        title=f"{symbol} Stock Price ({interval})",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series"
    )
    # To return KPIs and chart - orders must strictly match the Output (component_ids) as in the "@callback" above.
    return (
        f"{analysis_result['max_price']}",
        f"{analysis_result['min_price']}",
        f"{analysis_result['change_pct']}%",
        f"{analysis_result['mean_price']}",
        f"{analysis_result['variance']}",
        fig
    )

API_KEY = '822IWH5UUHLCZE77'            # API Key for Alpha Vantage
#for testing purposes

#we are defining a function call fetching stock data which takes in two inputs:
#symbol for the stock eg IBM and interval ie daily, monthly or weekly and return a Dataframe in form of a table
def get_stock_data(symbol: str, interval: str) -> pd.DataFrame | None:
    """
    fetch stock time series data from ALPHA VANTAGE API.

    :param symbol: (string): stock ticker symbol(eg 'IBM' or 'AAPL')
    :param interval: (string): time interval ('daily' or 'weekly')

    :return: pandas DataFrame or nothing:
        Dataframe containing closing price indexed by date
        None if the interval is invalid or retrieval failed due to other reasons.
    """
    function_map = {
        'daily': 'TIME_SERIES_DAILY',
        'weekly': 'TIME_SERIES_WEEKLY',
        'monthly': 'TIME_SERIES_MONTHLY',
    }
    function = function_map.get(interval.lower())
    if not function or not symbol:
        return None

# Build request URL and call API
# Data provided by Alpha Vantage (https://www.alphavantage.co)
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    try:
        # Alpha Vantage prepared its data having two top-level keys.
        # The first [0] is a meta data and the real informational data are from [1]
        time_series_key = list(data.keys())[1]   #picks the second key which holds the actual time series data(dates and prices)

        df = pd.DataFrame.from_dict(data[time_series_key], orient='index') #convert the time series datas into pandas

        df = df.rename(columns=lambda x: x.split(". ")[1].lower()) #cleaning the column names ie

        df.index = pd.to_datetime(df.index) #coverting date into proper date format

        df = df.sort_index() # sorting the data by dates preparing for futher analysis

        #keeping only the columns we want to use
        return df[['close']].astype(float) #changing the data into numbers

    except:
        return None #if anything goes wrong return nothing

#Stock Analysis(KPIs)
#defining a function that takes a df and return calculations
def calculating_KPIs_single_stock(df) -> dict:
    """
     Calculate simple KPIs and moving averages series (7 and 30) for a single stock.

     :param: a DataFrame with:
       - DateTime index (sorted ascending)
       - A 'closing' column of float prices

     :return: Returns a dictionary containing:
       - change_pct, max_price, min_price, mean_price, variance: rounded to 2 decimals
       - ma7_series: 7-period moving average (pandas Series)
       - ma30_series: 30-period moving average (pandas Series)
    """

    if df is None or df.empty:
        return {}

# Use the close-price series for all computations
    close_list = df['close']

# Calculate the change_pct
    last_close = close_list.iloc[-1] #most recent closing price
    prev_close = close_list.iloc[-2] #closing price a day before
    change_pct = (last_close - prev_close) / prev_close * 100 # % change between last two days

# Calculate the moving average series
    ma7_series = close_list.rolling(window=7).mean()
    ma30_series = close_list.rolling(window=30).mean()

# Calculate the max, min, mean, and variance
    max_price = close_list.max() # finding the highest closing price in the whole df
    min_price = close_list.min() # lowest closing price in the whole df
    mean_price = close_list.mean()
    variance = close_list.var()

#returning the dictionary of KPIs all rounded to 2 decimal places
    return {
        'change_pct': round(change_pct, 2),
        'ma7_series': ma7_series,
        'ma30_series': ma30_series,
        'max_price': round(max_price, 2),
        'min_price': round(min_price, 2),
        'mean_price': round(mean_price, 2),
        'variance': round(variance, 2)
    }

# App running
if __name__ == '__main__':
    app.run(debug = False)