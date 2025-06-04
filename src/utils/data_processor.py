from pandas import DataFrame
import numpy as np

def clean_data(data: DataFrame) -> DataFrame:
    """Cleans the stock data by removing NaN values and duplicates."""
    data = data.dropna()
    data = data.drop_duplicates()
    return data

def transform_data(data: DataFrame) -> DataFrame:
    """Transforms the stock data for analysis, e.g., normalizing prices."""
    data['normalized_price'] = (data['close'] - data['close'].min()) / (data['close'].max() - data['close'].min())
    return data

def prepare_data(data: DataFrame) -> DataFrame:
    """Prepares the data for analysis by selecting relevant columns."""
    return data[['date', 'normalized_price', 'volume']]

def calculate_moving_average(data: DataFrame, window: int) -> DataFrame:
    """Calculates the moving average for the stock prices."""
    data['moving_average'] = data['close'].rolling(window=window).mean()
    return data

def detect_outliers(data: DataFrame, threshold: float = 1.5) -> DataFrame:
    """Detects outliers in the stock data using the IQR method."""
    Q1 = data['close'].quantile(0.25)
    Q3 = data['close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return data[(data['close'] >= lower_bound) & (data['close'] <= upper_bound)]