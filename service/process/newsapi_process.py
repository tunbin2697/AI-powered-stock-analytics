import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib

# Force PyTorch backend for transformers to avoid TensorFlow conflicts
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

from transformers import AutoTokenizer, AutoModel
import torch
from service.crawl.newsapi import NewsapiCrawl
# from sentence_transformers import SentenceTransformer  # Commented out as requested

class NewsApiProcessor:
    """
    NewsAPI processor for XAI and TGNN++ integration.
    
    Processes news articles for:
    - DAVOTS text feature attribution (news sentiment impact)
    - ICFTS causal analysis (news-driven market events)
    - TGNN++ multi-modal input (text embeddings for graph neural network)
    - Flask dashboard sentiment analysis
    """
    
    def __init__(self, raw_data_path="data/raw/newsapi", processed_data_path="data/processed/daily"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.newsapi_crawler = NewsapiCrawl(base_path=raw_data_path)
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Initialize FinBERT for 768-dim embeddings as per lab requirements
        self.finbert_model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize_finbert(self):
        """Initialize FinBERT model for text embedding (768-dim)"""
        if self.tokenizer is None or self.model is None:
            print("🔄 Loading FinBERT model for 768-dim embeddings...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
                self.model = AutoModel.from_pretrained(self.finbert_model_name)
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ FinBERT loaded successfully on {self.device}")
            except Exception as e:
                print(f"❌ Failed to load FinBERT: {str(e)}")
                print("📝 Using fallback: sentence-transformers/all-MiniLM-L6-v2")

                # Commented out as requested
                # self.model = SentenceTransformer('all-MiniLM-L6-v2')
                # self.tokenizer = "sentence_transformer"
                raise e  # Re-raise the error instead of using fallback
    
    def load_news_articles(self, ticker, start, end):
        """Load news articles using NewsAPI crawler"""
        try:
            articles = self.newsapi_crawler.load_data(ticker=ticker, start=start, end=end)
            print(f"✅ Loaded {len(articles)} news articles for {ticker}")
            return articles
        except Exception as e:
            print(f"❌ Failed to load news articles: {str(e)}")
            return []
    
    def extract_text_features(self, articles):
        """Extract relevant text content from news articles"""
        processed_articles = []
        
        for article in articles:
            # Extract text fields
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Combine text fields for comprehensive analysis
            full_text = f"{title}. {description}. {content}"
            
            # Clean and preprocess text
            full_text = self.clean_text(full_text)
            
            if full_text.strip():  # Only include non-empty articles
                processed_articles.append({
                    'publishedAt': article.get('publishedAt', ''),
                    'title': title,
                    'description': description,
                    'full_text': full_text,
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'Unknown')
                })
        
        print(f"🔧 Processed {len(processed_articles)} articles with text content")
        return processed_articles
    
    def clean_text(self, text):
        """Clean and preprocess text for FinBERT analysis"""
        if not text or text == 'None':
            return ""
        
        # Basic text cleaning
        text = str(text).strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Truncate to reasonable length for FinBERT (512 tokens max)
        if len(text) > 2000:  # Rough character limit
            text = text[:2000] + "..."
            
        return text
    
    def generate_finbert_embeddings(self, processed_articles):
        """Generate FinBERT embeddings (768-dim) for news articles"""
        self.initialize_finbert()
        
        embeddings = []
        texts = []
        timestamps = []
        
        for article in processed_articles:
            text = article['full_text']
            timestamp = article['publishedAt']
            
            if not text.strip():
                continue
                
            try:
                # Commented out SentenceTransformer fallback as requested
                # if self.tokenizer == "sentence_transformer":
                #     # Fallback: SentenceTransformer
                #     embedding = self.model.encode(text)
                #     # Pad or truncate to 768 dimensions
                #     if len(embedding) < 768:
                #         embedding = np.pad(embedding, (0, 768 - len(embedding)))
                #     elif len(embedding) > 768:
                #         embedding = embedding[:768]
                # else:
                
                # FinBERT processing only
                inputs = self.tokenizer(text, return_tensors="pt", 
                                      truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding (768-dim)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                
                embeddings.append(embedding)
                texts.append(text)
                timestamps.append(timestamp)
                
            except Exception as e:
                print(f"⚠️ Failed to embed article: {str(e)[:100]}...")
                continue
        
        print(f"✅ Generated {len(embeddings)} FinBERT embeddings (768-dim)")
        return np.array(embeddings), texts, timestamps
    
    def aggregate_daily_embeddings(self, embeddings, timestamps):
        """Aggregate embeddings by day for time series alignment"""
        if len(embeddings) == 0:
            return np.array([]), []
        
        # Convert timestamps to dates
        daily_embeddings = {}
        
        for i, timestamp in enumerate(timestamps):
            try:
                date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                date_str = date.strftime('%Y-%m-%d');
                
                if date_str not in daily_embeddings:
                    daily_embeddings[date_str] = []
                daily_embeddings[date_str].append(embeddings[i])
            except:
                continue
        
        # Average embeddings per day
        aggregated_embeddings = []
        dates = []
        
        for date_str in sorted(daily_embeddings.keys()):
            day_embeddings = daily_embeddings[date_str]
            avg_embedding = np.mean(day_embeddings, axis=0)
            aggregated_embeddings.append(avg_embedding)
            dates.append(date_str)
        
        print(f"📅 Aggregated embeddings into {len(dates)} daily averages")
        return np.array(aggregated_embeddings), dates
    
    def process_raw_data(self, ticker, start, end, save_output=True):

        print(f"📰 Processing NewsAPI data for Step 7 (DAVOTS/ICFTS/TGNN++)...")
        
        # Load news articles using crawler
        articles = self.load_news_articles(ticker, start, end)
        
        if not articles:
            print("⚠️ No news articles found")
            return np.array([]), [], []
        
        # Extract and clean text features
        processed_articles = self.extract_text_features(articles)
        
        # Generate FinBERT embeddings (768-dim)
        embeddings, texts, timestamps = self.generate_finbert_embeddings(processed_articles)
        
        # Aggregate by day for time series alignment
        daily_embeddings, dates = self.aggregate_daily_embeddings(embeddings, timestamps)
        
        # Save processed data
        if save_output and len(daily_embeddings) > 0:
            self.save_processed_data(ticker, daily_embeddings, dates, texts)
        
        return daily_embeddings, dates, processed_articles
    
    def save_processed_data(self, ticker, embeddings, dates, texts):
        """Save processed news embeddings and metadata"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Save embeddings as .npy file (as per lab requirements)
        embeddings_file = f"processed_news_emb_{ticker}_{today}.npy"
        embeddings_path = os.path.join(self.processed_data_path, embeddings_file)
        np.save(embeddings_path, embeddings)
        
        print(f"✅ Saved news embeddings: {embeddings_file}")
        print(f"📂 Location: {self.processed_data_path}")
        
        return True
    
    def load_processed_embeddings(self, ticker, date_str=None):
        """Load saved news embeddings"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        embeddings_file = f"processed_news_emb_{ticker}_{date_str}.npy"
        embeddings_path = os.path.join(self.processed_data_path, embeddings_file)
        
        try:
            embeddings = np.load(embeddings_path)
            print(f"✅ Loaded news embeddings from {embeddings_file}")
            return embeddings
        except FileNotFoundError:
            print(f"❌ News embeddings not found: {embeddings_file}")
            return None
 
    def analyze_sentiment_finbert(self, processed_articles):
        """
        Analyze sentiment using FinBERT for DAVOTS visualization.
        Returns sentiment scores (negative: -1 to 0, neutral: 0, positive: 0 to 1)
        """
        print("🔄 Analyzing sentiment with FinBERT for DAVOTS...")
          # Initialize FinBERT for sentiment analysis (classification head)
        try:
            from transformers import pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                framework="pt"  # Explicitly use PyTorch framework
            )
            print("✅ FinBERT sentiment analyzer loaded with PyTorch backend")
        except Exception as e:
            print(f"⚠️ FinBERT sentiment failed: {str(e)}, using VADER fallback")
            # Fallback to VADER sentiment
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sentiment_pipeline = SentimentIntensityAnalyzer()
        
        sentiment_data = []
        
        for article in processed_articles:
            text = article['full_text']
            timestamp = article['publishedAt']
            
            if not text.strip():
                continue
                
            try:
                if hasattr(sentiment_pipeline, 'polarity_scores'):
                    # VADER sentiment (fallback)
                    scores = sentiment_pipeline.polarity_scores(text)
                    sentiment_score = scores['compound']  # Range: -1 to 1
                    sentiment_label = 'positive' if sentiment_score > 0.1 else ('negative' if sentiment_score < -0.1 else 'neutral')
                    confidence = abs(sentiment_score)
                else:
                    # FinBERT sentiment
                    result = sentiment_pipeline(text[:512])[0]  # Truncate for FinBERT
                    sentiment_label = result['label'].lower()
                    confidence = result['score']
                    
                    # Convert to numeric score for DAVOTS
                    if sentiment_label == 'positive':
                        sentiment_score = confidence
                    elif sentiment_label == 'negative':
                        sentiment_score = -confidence
                    else:  # neutral
                        sentiment_score = 0.0
                
                sentiment_data.append({
                    'timestamp': timestamp,
                    'title': article['title'][:100],
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'confidence': confidence,
                    'text_length': len(text)
                })
                
            except Exception as e:
                print(f"⚠️ Sentiment analysis failed for article: {str(e)[:100]}...")
                continue
        
        print(f"✅ Analyzed sentiment for {len(sentiment_data)} articles")
        return sentiment_data
    
    def aggregate_daily_sentiment(self, sentiment_data):
        """
        Aggregate sentiment scores by day for DAVOTS time series analysis.
        Returns daily sentiment scores and metadata for XAI attribution.
        """
        if not sentiment_data:
            return [], []
        
        daily_sentiment = {}
        
        for item in sentiment_data:
            try:
                date = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).date()
                date_str = date.strftime('%Y-%m-%d')
                
                if date_str not in daily_sentiment:
                    daily_sentiment[date_str] = {
                        'scores': [],
                        'articles': [],
                        'positive_count': 0,
                        'negative_count': 0,
                        'neutral_count': 0
                    }
                
                daily_sentiment[date_str]['scores'].append(item['sentiment_score'])
                daily_sentiment[date_str]['articles'].append(item['title'])
                
                # Count sentiment categories
                if item['sentiment_label'] == 'positive':
                    daily_sentiment[date_str]['positive_count'] += 1
                elif item['sentiment_label'] == 'negative':
                    daily_sentiment[date_str]['negative_count'] += 1
                else:
                    daily_sentiment[date_str]['neutral_count'] += 1
                    
            except:
                continue
        
        # Calculate daily aggregated metrics
        daily_data = []
        dates = []
        
        for date_str in sorted(daily_sentiment.keys()):
            day_data = daily_sentiment[date_str]
            scores = day_data['scores']
            
            # Calculate daily sentiment metrics for DAVOTS
            daily_metrics = {
                'date': date_str,
                'avg_sentiment': np.mean(scores),
                'sentiment_volatility': np.std(scores),
                'sentiment_range': max(scores) - min(scores),
                'positive_ratio': day_data['positive_count'] / len(scores),
                'negative_ratio': day_data['negative_count'] / len(scores),
                'article_count': len(scores),
                'sentiment_momentum': np.mean(scores) * len(scores),  # Weighted by volume
            }
            
            daily_data.append(daily_metrics)
            dates.append(date_str)
        
        print(f"📅 Aggregated sentiment into {len(dates)} daily metrics for DAVOTS")
        return daily_data, dates

    