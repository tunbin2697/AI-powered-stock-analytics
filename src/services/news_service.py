import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pygooglenews import GoogleNews
import datetime
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NewsService:
  
    def __init__(self):
        """
        Initializes the service by loading the sentiment analysis model and tokenizer.
        This ensures they are loaded only once.
        """
        print("Loading FinBERT model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("Model and tokenizer loaded.")
  
    @staticmethod
    def fetch_news_data(tickers, start_date_str, end_date_str):
        gn = GoogleNews(lang='en', country='US')

        all_news_data = []

        for ticker in tickers:
            try:
                news_results = gn.search(ticker, from_=start_date_str, to_=end_date_str)
                for entry in news_results.get('entries', []):
                    title = entry['title']
                    try:
                        published_str = entry['published']
                        published_date = datetime.datetime.strptime(published_str[:-4], "%a, %d %b %Y %H:%M:%S").date()
                    except ValueError:
                        continue
                    all_news_data.append({
                        'ticker': ticker,
                        'title': title,
                        'Date': published_date
                    })
            except Exception as e:
                print(f"Error fetching news for {ticker}: {e}")
                

        df = pd.DataFrame(all_news_data)

        if df.empty:
            # Create an empty DataFrame with all dates if no news data is found
            all_dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
            full_date_df = pd.DataFrame({'Date': all_dates})
            return full_date_df

        df_grouped = df.groupby(['Date', 'ticker'])['title'].apply(lambda titles: '\n'.join(titles)).reset_index()

        df_pivot = df_grouped.pivot(index='Date', columns='ticker', values='title')

        df_pivot.columns = [f"{query}_title" for query in df_pivot.columns]

        df_pivot = df_pivot.reset_index()

        df_pivot['Date'] = pd.to_datetime(df_pivot['Date'])

        return df_pivot
    

# news_raw_data = fetch_news_data(["AAPL", "TSLA"], "2020-01-01", "2025-01-01")
# display(news_raw_data)

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text using the pre-loaded FinBERT model.
        Returns sentiment scores (positive, negative, neutral).
        """
        if pd.isna(text):
            return {'positive': 0, 'negative': 0, 'neutral': 0}

        # Use the model and tokenizer loaded in __init__
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        predictions = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

        # Ensure predictions is a list, even for single-item output
        if not isinstance(predictions, list):
            predictions = [predictions]

        # Map scores to sentiment labels based on the model's output order
        # (assuming the default order is positive, negative, neutral)
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        if len(predictions) == 3: # Check if all three sentiment scores are present
            sentiment_scores['positive'] = predictions[0]
            sentiment_scores['negative'] = predictions[1]
            sentiment_scores['neutral'] = predictions[2]
        elif len(predictions) > 0: # Handle cases where output might be different (e.g., binary)
            # This part might need adjustment based on actual model output for non-standard cases
            # For typical FinBERT, it's 3 classes.
            print(f"Warning: Unexpected number of sentiment scores ({len(predictions)}). Assuming positive score is the first one.")
            sentiment_scores['positive'] = predictions[0]


        return sentiment_scores


    def sentiment_3_class_news_data(self, input_df):

        sentiment_data = input_df.copy()
        for col in sentiment_data.columns:
            if col.endswith('_title'):
                ticker = col.replace('_title', '')
                print(f"Analyzing sentiment for column: {col}")
                sentiment_results = sentiment_data[col].apply(lambda x: self.analyze_sentiment(x))

                sentiment_data[f'{ticker}_sentiment_positive'] = sentiment_results.apply(lambda x: x.get('positive', 0))
                sentiment_data[f'{ticker}_sentiment_negative'] = sentiment_results.apply(lambda x: x.get('negative', 0))
                sentiment_data[f'{ticker}_sentiment_neutral'] = sentiment_results.apply(lambda x: x.get('neutral', 0))

        sentiment_data = sentiment_data.drop(columns=[col for col in sentiment_data.columns if col.endswith('_title')])
        return sentiment_data
      
# sentimented_news_data = sentiment_3_class_news_data(news_raw_data)
# display(sentimented_news_data)
