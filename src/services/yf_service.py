import yfinance as yf
import pandas as pd
import numpy as np
import ta 

class YahooFinanceData:

    def __init__(self,):
        pass
        
    @staticmethod
    def fetch_yahoo_data(tickers, start_date, end_date, interval='1d'):
        """
        Fetch stock data from Yahoo Finance, ensuring all dates in range are present.
        Returns: DataFrame with all dates and NaNs for non-trading days
        """
        try:
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True  # Fix for the warning
            )
            df.columns = [f'{ticker}_{attr.lower()}_stock' for attr, ticker in df.columns]
            
            # Add ticker and reset index
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])

        except Exception as e:
            print(f"Error fetching {tickers}: {str(e)}")
            return None
        return df
    
# yf_raw_data = fetch_yahoo_data(["AAPL", "TSLA"], "2020-01-01", "2025-01-01")
# display(yf_raw_data)

    @staticmethod
    def featuring_stock_data(input_df, start_date, end_date):

        processed_df = input_df.copy()

        close_cols = [col for col in input_df.columns if col.endswith("_close_stock")]

        for close_col in close_cols:
            ticker = close_col.replace("_close_stock", "")
            prefix = f"{ticker}_"
            processed_df[close_col] = processed_df[close_col].fillna(processed_df[close_col].mean())
            close_series = processed_df[close_col]

            processed_df[prefix + 'close_stock'] = close_series
            processed_df[prefix + 'daily_return_stock'] = close_series.pct_change()

            if len(close_series) >= 30:
                processed_df[prefix + 'ma50_stock'] = close_series.rolling(window=50).mean()
                processed_df[prefix + 'volatility_stock'] = processed_df[prefix + 'daily_return_stock'].rolling(window=14).std()
                processed_df[prefix + 'lag1_stock'] = processed_df[prefix + 'daily_return_stock'].shift(1)
                processed_df[prefix + 'lag2_stock'] = processed_df[prefix + 'daily_return_stock'].shift(2)
                processed_df[prefix + 'rsi14_stock'] = ta.momentum.rsi(close_series, window=14)
                processed_df[prefix + 'rsi21_stock'] = ta.momentum.rsi(close_series, window=21)

            if len(close_series) >= 250:
                processed_df[prefix + 'ma200_stock'] = close_series.rolling(window=200).mean()

        return processed_df

# featured_stock_data = featuring_stock_data(yf_raw_data, "2020-01-01", "2025-01-01")
# display(featured_stock_data)


if __name__ == "__main__":
    # To test the functions, you must call them through the class
    # and provide the necessary arguments.
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    tickers = ["AAPL", "GOOGL"]

    # 1. Fetch the data
    raw_data = YahooFinanceData.fetch_yahoo_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )

    if raw_data is not None:
        print("--- Raw Data ---")
        print(raw_data.head())
    else:
        print("no data")