import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Initialize APIs - Store keys in environment variables
NEWS_API_KEY = os.getenv('NEWSAPI_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Sample sentiment words - replace with comprehensive lists
POSITIVE_WORDS = ['up', 'rise', 'high', 'good', 'great', 'positive', 'strong', 'bullish', 'gain']
NEGATIVE_WORDS = ['down', 'fall', 'low', 'bad', 'poor', 'negative', 'weak', 'bearish', 'loss']

# 1. Data Fetching Functions
def fetch_yahoo_data(tickers, start_date, end_date, interval='1d'):
    """
    Fetch stock data from Yahoo Finance
    Returns: Dict of DataFrames {ticker: df}
    """
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True  # Fix for the warning
            )
            # Add ticker and reset index
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            data[ticker] = df
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            data[ticker] = pd.DataFrame()
    return data

def fetch_news_data(query, start_date, end_date):
    """
    Fetch news articles from News API
    Returns: List of dicts with article metadata
    """
    newsapi_url = "https://newsapi.org/v2/everything"
    
    all_articles = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        try:
            params = {
                'q': query,
                'from': date_str,
                'to': date_str,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 100,
                'apiKey': NEWS_API_KEY
            }
            response = requests.get(newsapi_url, params=params)
            response.raise_for_status()
            news_data = response.json()
            
            for article in news_data.get('articles', []):
                # Handle missing fields
                article.setdefault('title', '')
                article.setdefault('description', '')
                article.setdefault('content', '')
                article.setdefault('publishedAt', date_str + 'T00:00:00Z')
                
                all_articles.append(article)
        except Exception as e:
            print(f"News API error for {date_str}: {str(e)}")
        current_date += timedelta(days=1)
    
    return all_articles

def fetch_fred_data(series_ids, start_date, end_date):
    """
    Fetch macroeconomic data from FRED
    Returns: DataFrame with datetime index
    """
    fred_url = "https://api.stlouisfed.org/fred/series/observations"
    data = pd.DataFrame()
    
    for series_id in series_ids:
        try:
            params = {
                'series_id': series_id,
                'observation_start': start_date,
                'observation_end': end_date,
                'api_key': FRED_API_KEY,
                'file_type': 'json'
            }
            response = requests.get(fred_url, params=params)
            response.raise_for_status()
            series_data = response.json()
            
            # Extract observations
            observations = []
            for obs in series_data.get('observations', []):
                try:
                    value = float(obs['value']) if obs['value'] != '.' else np.nan
                    observations.append({
                        'date': obs['date'],
                        series_id: value
                    })
                except:
                    continue
            
            # Create DataFrame for this series
            series_df = pd.DataFrame(observations)
            if not series_df.empty:
                series_df['date'] = pd.to_datetime(series_df['date'])
                series_df.set_index('date', inplace=True)
                data = data.join(series_df, how='outer') if not data.empty else series_df
            
        except Exception as e:
            print(f"Error fetching {series_id}: {str(e)}")
    
    return data

# 2. Data Processing Functions
def process_stock_data(raw_stock_data):
    """
    Process raw Yahoo Finance data
    Returns: DataFrame with processed features
    """
    processed_dfs = []
    for ticker, df in raw_stock_data.items():
        if df.empty:
            continue
            
        # Basic cleaning
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Handle missing date
        if 'date' not in df.columns:
            continue
            
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        
        # Technical indicators
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['volatility'] = df['daily_return'].rolling(window=14).std()
        
        # Handle missing data
        df = df.ffill().bfill()  # Fix for deprecation warning
        df.dropna(inplace=True)
        
        processed_dfs.append(df)
    
    return pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()

def process_news_data(raw_news, tickers):
    """
    Process news articles and generate embeddings
    Returns: DataFrame with date, sentiment, and entity count
    """
    # Process articles
    news_records = []
    for article in raw_news:
        # Handle None values
        title = article['title'] or ''
        description = article['description'] or ''
        content = f"{title} {description}"
        
        # Extract ticker mentions
        mentioned_tickers = []
        for ticker in tickers:
            if ticker.lower() in content.lower():
                mentioned_tickers.append(ticker)
        
        if not mentioned_tickers:
            continue
            
        # Simple sentiment analysis
        positive_count = sum(1 for word in POSITIVE_WORDS if word in content.lower())
        negative_count = sum(1 for word in NEGATIVE_WORDS if word in content.lower())
        sentiment = (positive_count - negative_count) / max(len(content.split()), 1)
        
        # Get publish date
        try:
            pub_date = pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d')
        except:
            pub_date = datetime.now().strftime('%Y-%m-%d')
        
        for ticker in mentioned_tickers:
            news_records.append({
                'date': pub_date,
                'ticker': ticker,
                'sentiment': sentiment,
                'entity_count': len(mentioned_tickers)
            })
    
    if not news_records:
        return pd.DataFrame()
    
    news_df = pd.DataFrame(news_records)
    
    # Aggregate by date and ticker
    grouped = news_df.groupby(['date', 'ticker']).agg({
        'sentiment': 'mean',
        'entity_count': 'sum'
    }).reset_index()
    
    return grouped

def process_macro_data(raw_macro_data):
    """
    Process FRED macroeconomic data
    Returns: Processed DataFrame
    """
    if raw_macro_data.empty:
        return pd.DataFrame()
    
    # Handle missing data
    imputer = KNNImputer(n_neighbors=3)
    macro_imputed = pd.DataFrame(
        imputer.fit_transform(raw_macro_data),
        index=raw_macro_data.index,
        columns=raw_macro_data.columns
    )
    
    # Calculate derivatives
    for col in macro_imputed.columns:
        macro_imputed[f'{col}_delta'] = macro_imputed[col].pct_change()
    
    # Normalization
    scaler = MinMaxScaler()
    macro_normalized = pd.DataFrame(
        scaler.fit_transform(macro_imputed),
        index=macro_imputed.index,
        columns=macro_imputed.columns
    )
    
    return macro_normalized.reset_index().rename(columns={'index': 'date'})

# 3. Data Merging Function
def merge_multi_modal_data(stock_df, news_df, macro_df, start_date, end_date):
    """
    Merge processed data from all sources into unified DataFrame
    Returns: Merged DataFrame indexed by date
    """
    # Create date range index
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    base_df = pd.DataFrame(index=date_range)
    
    if stock_df.empty:
        return base_df
    
    # Merge stock data
    stock_pivot = stock_df.pivot_table(
        index='date', 
        columns='Ticker', 
        values=['open', 'high', 'low', 'close', 'volume', 'daily_return', 'ma_7', 'ma_30', 'volatility'],
        aggfunc='first'
    )
    
    # Flatten multi-level columns
    stock_pivot.columns = [f"{col[1]}_{col[0]}" for col in stock_pivot.columns]
    merged_df = base_df.join(stock_pivot)
    
    # Merge news data
    if not news_df.empty:
        news_pivot = news_df.pivot_table(
            index='date', 
            columns='ticker', 
            values=['sentiment'],
            aggfunc='mean'
        )
        news_pivot.columns = [f"{col[1]}_sentiment" for col in news_pivot.columns]
        merged_df = merged_df.join(news_pivot)
    
    # Merge macro data
    if not macro_df.empty:
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        macro_df.set_index('date', inplace=True)
        merged_df = merged_df.join(macro_df)
    
    # Handle final missing values
    merged_df = merged_df.ffill().bfill()
    
    return merged_df

# 4. XAI Visualization Preparation
def prepare_xai_data(merged_df, target_ticker, lookback=30):
    """
    Prepare data for XAI visualizations (DAVOTS and ICFTS)
    Returns: Processed data structures
    """
    if merged_df.empty:
        return {
            'davots_matrix': np.array([]),
            'davots_columns': [],
            'davots_dates': [],
            'icfts_baseline': {}
        }
    
    # Filter columns for target ticker
    ticker_cols = [col for col in merged_df.columns if target_ticker in col]
    macro_cols = [col for col in merged_df.columns if any(
        k in col for k in ['FEDFUNDS', 'GDP', 'CPIAUCSL']
    )]
    all_cols = ticker_cols + macro_cols
    
    # Create DAVOTS-ready matrix
    davots_data = merged_df[all_cols].tail(lookback)
    
    # Prepare ICFTS counterfactual baseline
    baseline = davots_data.mean().to_dict() if not davots_data.empty else {}
    
    return {
        'davots_matrix': davots_data.values if not davots_data.empty else np.array([]),
        'davots_columns': davots_data.columns.tolist(),
        'davots_dates': davots_data.index.strftime('%Y-%m-%d').tolist(),
        'icfts_baseline': baseline
    }

# 5. T-GNN++ Training Preparation
def prepare_tgnn_data(merged_df, tickers, lookback=30, forecast_horizon=5):
    """
    Prepare data for T-GNN++ Transformer training
    Returns: Dict of tensors for model input
    """
    if merged_df.empty or not tickers:
        return {
            'X': np.array([]),
            'y': np.array([]),
            'asset_names': [],
            'feature_names': [],
            'timestamps': []
        }
    
    # Create 3D tensor: [days, assets, features]
    features = ['close', 'volume', 'daily_return', 'volatility', 'sentiment']
    
    data = []
    for ticker in tickers:
        ticker_data = []
        for feature in features:
            col_name = f"{ticker}_{feature}"
            if col_name in merged_df.columns:
                ticker_data.append(merged_df[col_name].values)
            else:
                # Fill with zeros if feature missing
                ticker_data.append(np.zeros(len(merged_df)))
        
        # Stack features for this ticker
        data.append(np.stack(ticker_data, axis=1))
    
    # Stack to create [days, assets, features] tensor
    tensor_3d = np.stack(data, axis=1)
    
    # Create sequences for training
    X, y = [], []
    for i in range(lookback, len(tensor_3d) - forecast_horizon):
        X.append(tensor_3d[i-lookback:i])
        y.append(tensor_3d[i+forecast_horizon, :, 0])  # Predict close price
    
    return {
        'X': np.array(X) if X else np.array([]),
        'y': np.array(y) if y else np.array([]),
        'asset_names': tickers,
        'feature_names': features,
        'timestamps': merged_df.index[lookback:-forecast_horizon].strftime('%Y-%m-%d').tolist()
    }

# Main Execution
if __name__ == "__main__":
    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'TSLA']
    START_DATE = '2025-06-05'
    END_DATE = '2025-07-03'
    FRED_SERIES = ['FEDFUNDS', 'GDP', 'CPIAUCSL']
    
    print("Starting data pipeline...")
    
    # 1. Fetch raw data
    print("\n[1/5] Fetching Yahoo Finance data...")
    raw_stock = fetch_yahoo_data(TICKERS, START_DATE, END_DATE)
    print(f"Retrieved {sum(len(df) for df in raw_stock.values())} stock records")
    
    print("\n[2/5] Fetching News API data...")
    raw_news = fetch_news_data(" OR ".join(TICKERS), START_DATE, END_DATE)
    print(f"Retrieved {len(raw_news)} news articles")
    
    print("\n[3/5] Fetching FRED data...")
    raw_macro = fetch_fred_data(FRED_SERIES, START_DATE, END_DATE)
    print(f"Retrieved {raw_macro.shape[0]} macro records")
    
    # 2. Process data
    print("\n[4/5] Processing data...")
    processed_stock = process_stock_data(raw_stock)
    print(f"Processed {len(processed_stock)} stock records")
    
    processed_news = process_news_data(raw_news, TICKERS)
    print(f"Processed {len(processed_news)} news records" if not processed_news.empty else "No news records processed")
    
    processed_macro = process_macro_data(raw_macro)
    print(f"Processed {len(processed_macro)} macro records" if not processed_macro.empty else "No macro records processed")
    
    # 3. Merge data
    print("\n[5/5] Merging multi-modal data...")
    merged_data = merge_multi_modal_data(
        processed_stock, 
        processed_news, 
        processed_macro,
        START_DATE,
        END_DATE
    )
    
    # Save merged data
    if not merged_data.empty:
        merged_data.to_csv("multi_modal_financial_data.csv")
        print(f"Saved merged data ({merged_data.shape[0]} days, {merged_data.shape[1]} features) to multi_modal_financial_data.csv")
    else:
        print("No data to save")
    
    # 4. Prepare XAI data
    if not merged_data.empty:
        print("\nPreparing XAI visualization data...")
        xai_data = prepare_xai_data(merged_data, 'TSLA')
        with open("xai_visualization_data.json", "w") as f:
            json.dump(xai_data, f)
        print(f"Saved XAI data for {len(xai_data['davots_dates'])} days")
    else:
        print("Skipping XAI preparation - no data")
    
    # 5. Prepare T-GNN training data
    if not merged_data.empty and TICKERS:
        print("\nPreparing T-GNN++ training data...")
        tgnn_data = prepare_tgnn_data(merged_data, TICKERS)
        if tgnn_data['X'].size > 0:
            np.savez("tgnn_training_data.npz", X=tgnn_data['X'], y=tgnn_data['y'])
            print(f"Saved T-GNN data: {tgnn_data['X'].shape[0]} samples")
        else:
            print("No T-GNN training data generated")
    else:
        print("Skipping T-GNN preparation - no data or tickers")
    
    print("\nPipeline complete!")

    # Output structure summary
    print("\nData Structure Summary:")
    if not merged_data.empty:
        print(f"- Merged DataFrame: {merged_data.shape} (dates × features)")
    if 'davots_matrix' in locals() and xai_data['davots_matrix'].size > 0:
        print(f"- DAVOTS Matrix: {xai_data['davots_matrix'].shape} (days × features)")
    if 'tgnn_data' in locals() and tgnn_data['X'].size > 0:
        print(f"- T-GNN Input: X={tgnn_data['X'].shape}, y={tgnn_data['y'].shape}")