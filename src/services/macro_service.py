import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

class MacroService:
  @staticmethod
  def fetch_fred_data(series_ids, start_date, end_date):
    """
    Fetch multiple FRED series and align them on a shared date index.
    Missing values for series that don't exist on certain dates will remain as NaN.
    """

    fred_url = "https://api.stlouisfed.org/fred/series/observations"
    all_series = []

    date_sets = []

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
                    observations.append({'Date': obs['date'], f"{series_id}_macro": value})
                except:
                    continue

            series_df = pd.DataFrame(observations)
            series_df['Date'] = pd.to_datetime(series_df['Date'])

            all_series.append(series_df)
            date_sets.append(set(series_df['Date']))

        except Exception as e:
            print(f"Failed to fetch {series_id}: {e}")
            continue

    # Build master date index (union of all dates)
    all_dates = sorted(set.union(*date_sets))
    master_index = pd.DataFrame({'Date': pd.to_datetime(all_dates)})

    # Align all series to master date
    aligned_series = []
    for series_df in all_series:
        aligned = master_index.merge(series_df, on='Date', how='left')
        aligned_series.append(aligned)

    # Merge all aligned series into final DataFrame
    from functools import reduce
    final_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), aligned_series)
    final_df.sort_values('Date', inplace=True)
    return final_df

  @staticmethod
  def ffill_macro_data(input_df):
    """
    Process macroeconomic data: forward fill missing values for columns ending with _macro.
    Returns: Processed DataFrame
    """
    ffill_df = input_df.copy()

    macro_cols = [col for col in ffill_df.columns if col.endswith('_macro')]

    ffill_df[macro_cols] = ffill_df[macro_cols].ffill()

    return ffill_df

# ffilled_macro_data = ffill_macro_data(fred_raw_data)
# display(ffilled_macro_data)