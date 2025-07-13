import pandas as pd

class DataPrepareService:
  
    @staticmethod
    def merge_and_fill_nan(stock_df, news_df, macro_df, date_col='Date', rolling_window=3):
      """
      Merges stock, news, and macro DataFrames on 'Date' (outer join),
      and fills NaNs based on column type:
        - '_stock' → rolling mean fill
        - '_sentiment_' → fill with 0
        - '_macro' → one-step forward fill only

      Returns:
          pd.DataFrame with merged and filled data.
      """
      # Step 1: Outer merge
      stock_df = stock_df.copy()
      news_df = news_df.copy()
      macro_df = macro_df.copy()

      stock_df[date_col] = pd.to_datetime(stock_df[date_col])
      news_df[date_col] = pd.to_datetime(news_df[date_col])
      macro_df[date_col] = pd.to_datetime(macro_df[date_col])

      merged_df = pd.merge(stock_df, news_df, on=date_col, how='outer')
      merged_df = pd.merge(merged_df, macro_df, on=date_col, how='outer')
      merged_df.sort_values(by=date_col, inplace=True)
      merged_df.reset_index(drop=True, inplace=True)

      # Step 2: Fill based on column types
      for col in merged_df.columns:
          if col == date_col:
              continue

          if col.endswith('_stock'):
              merged_df[col] = merged_df[col].fillna(
                  merged_df[col].rolling(window=rolling_window, min_periods=1, center=True).mean()
              )
              # Fallback for edge cases
              merged_df[col] = merged_df[col].ffill().bfill()

          elif '_sentiment_' in col:
              merged_df[col] = merged_df[col].fillna(0)

          elif col.endswith('_macro'):
              merged_df[col] = merged_df[col].fillna(method='ffill')

      return merged_df

# visualize_df = merge_and_fill_nan(featured_stock_data ,sentimented_news_data, ffilled_macro_data)
# display(visualize_df)

    @staticmethod
    def merge_and_fill_nan_no_news(stock_df, macro_df, date_col='Date', rolling_window=3):
        """
        Merges stock, news, and macro DataFrames on 'Date' (outer join),
        and fills NaNs based on column type:
          - '_stock' → rolling mean fill
          - '_sentiment_' → fill with 0
          - '_macro' → one-step forward fill only

        Returns:
            pd.DataFrame with merged and filled data.
        """

        stock_df[date_col] = pd.to_datetime(stock_df[date_col])
        macro_df[date_col] = pd.to_datetime(macro_df[date_col])

        merged_df = pd.merge(stock_df, macro_df, on=date_col, how='outer')
        merged_df.sort_values(by=date_col, inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        # Step 2: Fill based on column types
        for col in merged_df.columns:
            if col == date_col:
                continue

            if col.endswith('_stock'):
                merged_df[col] = merged_df[col].fillna(
                    merged_df[col].rolling(window=rolling_window, min_periods=1, center=True).mean()
                )
                # Fallback for edge cases
                merged_df[col] = merged_df[col].ffill().bfill()

            elif col.endswith('_macro'):
                merged_df[col] = merged_df[col].fillna(method='ffill')

        return merged_df
      
# merged_no_news = merge_and_fill_nan_no_news(featured_stock_data, ffilled_macro_data)
# display(merged_no_news)     

    @staticmethod   
    def merge_and_fill_nan_stock_only(stock_df, rolling_window=3):
        merged_df = stock_df.copy()
        for col in merged_df.columns:
            if col == "Date":
                continue
            if col.endswith('_stock'):
                merged_df[col] = merged_df[col].fillna(
                    merged_df[col].rolling(window=rolling_window, min_periods=1, center=True).mean()
                )
                merged_df[col] = merged_df[col].ffill().bfill()

        return merged_df
      
# stock_only_df = merge_and_fill_nan_stock_only(featured_stock_data)
# display(stock_only_df)


# you can chose the merged_df as which df you want
# merged_df = visualize_df
# merged_df = merged_no_news
# merged_df = stock_only_df