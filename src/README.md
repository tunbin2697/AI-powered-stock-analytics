# Financial Analysis Dashboard with AI & Chatbot

---

## 1. Main Concept

This project is a **Streamlit web application** that demonstrates a complete AI pipeline for financial time series analysis and prediction.  
It covers every step:

- **Data Acquisition:** Fetching stock, news, and macroeconomic data.
- **Data Processing:** Cleaning, merging, and aligning data.
- **Feature Engineering:** Creating technical and sentiment features.
- **EDA & Visualization:** Interactive charts for exploration and insight.
- **Model Training:** Multiple models (Linear Regression, Random Forest, LSTM, ARIMA) for stock price prediction.
- **Chatbot Integration:** An AI-powered chatbot (Gemini via LangChain) for interactive Q&A about any chart, data, or model result.

All steps are modular, with each service implemented as a Python class in `src/services/`, and orchestrated in `src/app.py`.  
The chatbot logic is in `src/chatbot.py`, and all user interaction is managed through the Streamlit UI.

**Code Example:**

```python
# filepath: [app.py]
from services.yf_service import YahooFinanceData
from services.news_service import NewsService
from services.macro_service import MacroService
from services.data_prepare_service import DataPrepareService
from services.visualize_service import VisualizeService
from services.linear_regression_service import LinearRegressionService
from services.random_forest_service import RandomForestService
from services.lstm_service import LSTMService
from services.arima_service import ARIMAService
from chatbot import ChatBot

yf_service = YahooFinanceData()
news_service = NewsService()
macro_service = MacroService()
prepare_service = DataPrepareService()
visualize_service = VisualizeService()
linear_regression_service = LinearRegressionService()
random_forest_service = RandomForestService()
lstm_service = LSTMService()
arima_service = ARIMAService()
chatbot = ChatBot()
```

---

## 2. Business Problem

### Definition & Reasoning

**Challenge:**

- AI and data science education lacks interactive, real-world tools.
- Stock prediction is a classic AI problem, but most solutions are either too technical or not user-friendly.
- Investors and students need a simple, hands-on way to explore, visualize, and predict stock prices.

**Purpose:**

- Make AI concepts accessible through real financial data.
- Enable users to interactively explore, visualize, and predict stock prices.
- Provide transparency and reproducibility in the AI pipeline.
- Offer a chatbot for natural language questions about data, models, and results.

### Approach

- **Modular pipeline:** Each step (data, features, models, chatbot) is a dedicated service.
- **Interactive UI:** Users select tickers, time ranges, and models, and ask questions about any chart or result.
- **AI Integration:** Chatbot uses Gemini (Google Generative AI) via LangChain for reliable, context-aware answers.

---

## 3. Data Preparation

### Data Sources

- **Stock Data:** Fetched from Yahoo Finance via `yf_service.py` for selected tickers (e.g., AAPL, TSLA).

  The `YahooFinanceData.fetch_yahoo_data` method uses the `yfinance` library to download historical stock data for any list of tickers and a specified date range. The raw data from Yahoo Finance is returned as a multi-index DataFrame, where columns are tuples like `('AAPL', 'Close')`, `('TSLA', 'Volume')`, etc.

  The code then **renames these columns** to a unified format such as `AAPL_close_stock`, `TSLA_volume_stock`, etc., making it easy to merge and process multiple tickers together. This is done by:

  ```python
  # After download, columns are multi-index: (attribute, ticker)
  df.columns = [f'{ticker}_{attr.lower()}_stock' for attr, ticker in df.columns]
  ```

  The function also **resets the index** and ensures a `Date` column is present and in datetime format:

  ```python
  df.reset_index(inplace=True)
  df['Date'] = pd.to_datetime(df['Date'])
  ```

  If multiple tickers are requested, the resulting DataFrame contains all requested tickers as separate columns, aligned by date. Non-trading days are filled with NaNs, so all dates in the range are present for all tickers, which is important for time series modeling and merging with other datasets.

  **Example usage:**

  ```python
  yf_raw_data = YahooFinanceData.fetch_yahoo_data(
      tickers=["AAPL", "TSLA"],
      start_date="2020-01-01",
      end_date="2025-01-01"
  )
  # Resulting columns: Date, AAPL_close_stock, TSLA_close_stock, AAPL_volume_stock, TSLA_volume_stock, ...
  ```

  This unified structure allows for easy feature engineering and merging with news and macro data later in the pipeline.

- **News Data:** Headlines and articles for tickers and market topics via `news_service.py`.

  The `NewsService.fetch_news_data` method uses the `pygooglenews` library to collect news headlines for each selected ticker and date range. For each ticker, it searches Google News and aggregates all headlines published on each day. The resulting DataFrame has columns like `AAPL_title`, `TSLA_title`, etc., with each cell containing all headlines for that ticker and date.

  The code then **pivots and aligns the news data** so that each row corresponds to a date, and each column to a ticker's news headlines. Missing dates are filled so the news data aligns with stock and macro data.

  **Example usage:**

  ```python
  news_raw_data = NewsService.fetch_news_data(
      tickers=["AAPL", "TSLA"],
      start_date_str="2020-01-01",
      end_date_str="2025-01-01"
  )
  # Resulting columns: Date, AAPL_title, TSLA_title, ...
  ```

  **Example code:**

  ```python
  sentimented_news_data = news_service.sentiment_3_class_news_data(news_raw_data)
  # Resulting columns: Date, AAPL_sentiment_positive, AAPL_sentiment_negative, AAPL_sentiment_neutral, ...
  ```

  This process transforms raw news headlines into structured, date-aligned raw title features for each ticker, ready to be sentimented

- **Macroeconomic Data:** Indicators (e.g., GDP, CPI, FEDFUNDS) from FRED via `macro_service.py`.

  The `MacroService.fetch_fred_data` method retrieves raw macroeconomic time series data from the FRED API for each selected indicator and date range. For each indicator (e.g., "GDP", "FEDFUNDS"), it sends a request to FRED and receives a JSON response containing daily or monthly observations.

  The raw data for each indicator is loaded into a DataFrame with columns:

  - `Date`: The observation date.
  - `{indicator}_macro`: The value for that indicator on that date.

  Each indicator is fetched separately, and the code aligns all series on a **master date index** (the union of all dates across all indicators). This ensures that every date in the range is present, with missing values as NaN if an indicator is not reported on that date.

  **Example code from the codebase:**

  ```python
  fred_raw_data = MacroService.fetch_fred_data(
      series_ids=["FEDFUNDS", "GDP", "CPIAUCSL"],
      start_date="2020-01-01",
      end_date="2025-01-01"
  )
  # Resulting columns: Date, FEDFUNDS_macro, GDP_macro, CPIAUCSL_macro
  # Each column contains the raw value for that indicator, aligned by date.
  ```

  The initial format is a wide DataFrame with one column per indicator, and each row is a date. Missing values are left as NaN for further processing. This raw macro data is saved for reproducibility and later merged with stock and news data after forward-filling missing values.

  This approach ensures that macroeconomic context is available for every date in the modeling pipeline, even if some indicators are only reported monthly or have missing entries.

### Data Handling

- **Processing Stock Data:** Raw data saved in CSV files for reproducibility (`Config.RAW_DATA_DIR`).

  The `YahooFinanceData` service handles raw stock data acquisition and feature engineering:

  **Feature Engineering:**

  - The `featuring_stock_data` method takes the raw DataFrame and computes a variety of technical features for each ticker:
    - **Daily Return:** Percentage change in close price.
    - **Moving Averages:** MA50 and MA200 for trend detection.
    - **Volatility:** Rolling standard deviation of daily returns.
    - **Lag Features:** Previous day and two-day lagged returns.
    - **RSI:** Relative Strength Index (14 and 21 days) for momentum.
  - These features are added as new columns, following the same naming convention (`AAPL_ma50_stock`, `TSLA_volatility_stock`, etc.).
  - Missing values are filled using mean or rolling statistics, ensuring the feature set is complete for modeling.

  **Code Example:**

  ```python
  featured_stock_data = YahooFinanceData.featuring_stock_data(raw_data, "2020-01-01", "2025-01-01")
  # Resulting columns: Date, AAPL_close_stock, AAPL_daily_return_stock, AAPL_ma50_stock, AAPL_volatility_stock, ...
  ```

  **How This Helps Future Steps:**

  - The unified, feature-rich DataFrame allows for easy merging with news and macro data.
  - All features are aligned by date and ticker, supporting time series modeling and multi-ticker analysis.
  - Technical indicators and lagged features improve predictive power for downstream models (regression, LSTM, TGNN, etc.).
  - Saving raw and processed data to CSV ensures reproducibility and transparency in the pipeline.

  **In summary:**  
  The `yf_service.py` module transforms raw Yahoo Finance data into a structured, feature-rich format, ready for merging, visualization, and machine

- **Processing News Data:**
  After fetching, the raw news data is processed to extract sentiment features for each ticker and date using FinBERT.

  **What is FinBERT?**  
   FinBERT is a transformer-based NLP model pre-trained specifically for financial sentiment analysis. It can classify financial text (such as news headlines) into positive, negative, or neutral sentiment classes. This makes it highly suitable for extracting market sentiment from news data.

  **How is FinBERT used in the code?**  
   The `NewsService` class loads FinBERT once in its constructor:

  ```python
  self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
  self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
  ```

  For each news headline, the `analyze_sentiment` method applies FinBERT and returns a dictionary of sentiment scores:

  ```python
  tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
  with torch.no_grad():
      outputs = self.model(**tokens)
  predictions = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
  sentiment_scores = {'positive': predictions[0], 'negative': predictions[1], 'neutral': predictions[2]}
  ```

  **How is the DataFrame constructed and formatted?**  
   The raw news DataFrame has columns like `Date`, `AAPL_title`, `TSLA_title`, etc., where each cell contains all headlines for that ticker and date.  
   The `sentiment_3_class_news_data` method applies FinBERT to each headline column, creating new columns for each ticker and sentiment class:

  - `{ticker}_sentiment_positive`
  - `{ticker}_sentiment_negative`
  - `{ticker}_sentiment_neutral`

  The original `_title` columns are dropped, leaving only the sentiment scores and the `Date` column.  
   The resulting DataFrame is fully aligned by date, with missing values filled as zeros.

  **Code Example from the codebase:**

  ```python
  sentimented_news_data = news_service.sentiment_3_class_news_data(news_raw_data)
  # Resulting columns: Date, AAPL_sentiment_positive, AAPL_sentiment_negative, AAPL_sentiment_neutral, ...
  ```

  **Why this approach?**

  - FinBERT provides domain-specific sentiment analysis, which is more accurate for financial news than general-purpose models.
  - The structured sentiment features allow downstream models to incorporate market mood and news impact for each ticker and date.
  - The format ensures easy merging with stock and macro data, supporting time series modeling and multi-ticker analysis.

  **In summary:**  
   Raw news headlines are transformed into structured, date-aligned sentiment features using FinBERT, enabling robust integration of news sentiment into the financial modeling pipeline.

- **Processing Macro Data:**
  After fetching, the raw macroeconomic data often contains missing values (NaNs) because some indicators are only reported monthly or have gaps in the FRED database. To ensure every date in the modeling pipeline has macro context, the code applies **forward fill** to all macro columns.

  **How is forward fill applied?**
  The `MacroService.ffill_macro_data` method takes the raw macro DataFrame and fills missing values forward for all columns ending with `_macro`:

  ```python
  ffill_df = input_df.copy()
  macro_cols = [col for col in ffill_df.columns if col.endswith('_macro')]
  ffill_df[macro_cols] = ffill_df[macro_cols].ffill()
  return ffill_df
  ```

  This ensures that for any date where a macro indicator is missing, the most recent available value is used instead.

  **Why forward fill?**

  - Most macroeconomic indicators (like GDP, CPI, interest rates) change slowly and are reported infrequently.
  - Forward filling allows models to use the latest known value until a new one is reported, which reflects how these indicators are used in real-world financial analysis.
  - This avoids gaps in the feature set and ensures all dates are aligned for merging with stock and news data.

  **Code Example:**

  ```python
  ffilled_macro_data = MacroService.ffill_macro_data(fred_raw_data)
  # All macro columns now have no missing values for any date in the modeling range.
  ```

  **Meaning:**

  - The processed macro DataFrame provides a complete, date-aligned set of economic features for every day in the analysis period.
  - This enables robust integration of macro context into time series models and ensures no missing values disrupt training

- **Merging and Filling NaNs:**  
  After cleaning and aligning each dataset, the final step is to merge stock, news, and macro data into a single DataFrame using `prepare_service.merge_and_fill_nan`. This step is crucial because each source may have missing dates or values due to market holidays, reporting frequency, or unavailable news.

  **Why is this necessary?**

  - When merging datasets by date, some dates may be present in one source but missing in another, resulting in NaN values.
  - Models require complete, aligned data for every time step; missing values can break training or reduce accuracy.
  - Filling NaNs ensures that every feature is available for every date, enabling robust time series modeling.

  **How is the final DataFrame constructed?**

  - The merge is performed on the `Date` column, using an outer join to include all dates from all sources.
  - After merging, the code fills NaNs using different strategies:
    - **Stock features:** Filled using rolling mean and forward/backward fill to preserve trends.
    - **Sentiment features:** Filled with zeros, since missing news means neutral sentiment.
    - **Macro features:** Forward-filled, as macro indicators change slowly and are reported infrequently.

  **Trade-offs:**

  - Filling NaNs can introduce bias if too many values are missing, but is necessary for model stability.
  - Forward-filling macro data reflects real-world usage, but may not capture sudden economic changes.
  - Rolling mean for stock features smooths out gaps but may dilute sharp market moves.

  **Code Example from the codebase:**

  ```python
  final_df = prepare_service.merge_and_fill_nan(
      featured_stock_data, sentimented_news_data, ffilled_macro_data
  )
  # Resulting columns: Date, AAPL_close_stock, TSLA_close_stock, ..., AAPL_sentiment_positive, ..., FEDFUNDS_macro, ...
  # All features are aligned by date, with no missing values.
  ```

  **Format:**

  - The final DataFrame is wide, with one row per date and columns for every feature from each source.
  - All missing values are filled, ensuring the data is ready for feature engineering, visualization, and modeling.

  **In summary:**  
  Merging and filling NaNs is a critical step to create a complete, date-aligned dataset for financial modeling, balancing data integrity and

## 4. Data Visualization

### Need & Reasoning

- **Understanding:** Visualizations help users grasp data distributions, trends, and relationships.
- **Communication:** Charts make results accessible and interpretable for all users.

### Explanation of Each Chart

- **EDA Summary:**

  - **What:** Displays basic statistics (mean, median, min, max, missing values) for all features in the final DataFrame.
  - **Purpose:** Gives users a quick overview of the data distribution and quality.
  - **How:** Uses `quick_eda(final_df)` to summarize columns and visualize distributions.

- **Correlation Heatmap:**

  - **What:** Shows pairwise correlation coefficients between all numerical features.
  - **Purpose:** Helps identify relationships, multicollinearity, and feature redundancy.
  - **How:** `visualize_service.create_correlation_heatmap(final_df)` computes and plots a heatmap using Plotly.

- **OHLCV Chart:**

  - **What:** Candlestick chart with volume bars for a selected ticker.
  - **Purpose:** Visualizes price movements and trading volume over time.
  - **How:** `visualize_service.create_ohlcv_fig(final_df, ticker)` builds a Plotly candlestick and bar chart.

- **Returns Histogram:**

  - **What:** Histogram of daily returns for a selected stock.
  - **Purpose:** Shows volatility, risk, and the distribution of returns.
  - **How:** `visualize_service.create_daily_return_histogram(final_df, ticker)` plots the histogram.

- **Sentiment Line Chart:**

  - **What:** Line chart of positive, negative, and neutral sentiment scores over time for a ticker.
  - **Purpose:** Tracks market mood and news impact.
  - **How:** `visualize_service.create_sentiment_line_chart(final_df, ticker)` plots sentiment scores.

- **Word Cloud:**

  - **What:** Visual representation of the most frequent words in news headlines for a ticker.
  - **Purpose:** Highlights trending topics and themes in news coverage.
  - **How:** `visualize_service.create_news_wordcloud_figure(news_raw_data, ticker)` generates a word cloud using the WordCloud library.

- **Macro Timeseries:**

  - **What:** Line chart of macroeconomic indicators (e.g., FEDFUNDS, GDP) over time.
  - **Purpose:** Shows economic trends and context for market analysis.
  - **How:** `visualize_service.create_macro_timeseries_line_chart(final_df, indicator)` plots the selected macro series.

- **RSI & MA Charts:**

  - **What:** Line charts for Relative Strength Index (RSI) and Moving Averages (MA50, MA200) for each ticker.
  - **Purpose:** Reveal technical signals for overbought/oversold conditions and trend direction.
  - **How:** `visualize_service.create_rsi_figs(final_df)` and `visualize_service.create_ma_figs(final_df)` plot these indicators.

- **Missing Value Bar Chart:**
  - **What:** Bar chart showing the count and percentage of missing values per column.
  - **Purpose:** Diagnoses data quality and highlights columns needing attention.
  - **How:** `visualize_service.create_missing_value_bar_chart(final_df)` plots missing value statistics.

---

\*\*In the app, each chart is interactive and can be used to explore, diagnose, and communicate insights about the data.

---

## 6. Model Services

### Model Services: Detailed Workflow & Improvements

#### Linear Regression

- **Data Handling:**

  - Features are scaled using `StandardScaler` and split into train/test sets in time order (no shuffling), preserving the time series structure.
  - Only non-target columns are used as features, ensuring no leakage.

- **Training & Saving:**

  - The model is trained on the scaled training set and saved with its scaler for reproducibility.
  - Improved by saving both the model and scaler, so inference uses the same scaling as training.

  ```python
  X_tr, X_te, y_tr, y_te, scaler, dates_te = linear_regression_service.process_data(final_df, target_ticker, is_training=True)
  linear_regression_service.train_and_save_model(X_tr, y_tr, scaler, model_path, scaler_path)
  ```

- **Prediction & Inference:**

  - For test prediction, the model loads the scaler and applies it to the test set.
  - For next-day inference, the latest row is scaled and predicted.

  ```python
  y_pred, y_true, dates = linear_regression_service.load_and_predict(final_df, target_ticker, model_path, scaler_path, is_test=True)
  pred_price, pred_date = linear_regression_service.load_and_predict(final_df, target_ticker, model_path, scaler_path, is_test=False)
  ```

- **Plotting:**

  - Actual vs predicted prices are plotted with R² score for evaluation.

  ```python
  fig = linear_regression_service.create_actual_vs_predicted_figure(y_true, y_pred, dates)
  ```

- **Improvement:**
  - Time-series split avoids lookahead bias, and saving the scaler ensures consistent feature scaling for inference.

---

#### Random Forest

- **Data Handling:**

  - Predicts next-day price change (`diff`) rather than absolute price, which helps the model focus on short-term movements and reduces non-stationarity.
  - Features are scaled, and train/test split is time-ordered.

  ```python
  X_train_scaled, X_test_scaled, y_train_diff, y_test_diff, scaler_X, date_test, y_test_actual = RandomForestService.process_rf_features(merged_df, target_code, is_training=True)
  ```

- **Training & Saving:**

  - Trains a `RandomForestRegressor` and saves both the model and scaler.
  - Improved by using price difference as target, which is easier for tree models to learn.

  ```python
  RandomForestService.train_and_save_rf_model(X_train_scaled, y_train_diff, scaler_X)
  ```

- **Prediction & Inference:**

  - For test prediction, reconstructs predicted price from predicted difference and actual price.
  - For next-day inference, adds predicted difference to last known price.

  ```python
  y_pred_prices, y_actual_prices, date_series = rf_service.load_rf_model_and_predict(merged_df, target_code, is_test=True)
  predicted_price = rf_service.load_rf_model_and_predict(merged_df, target_code, is_test=False)
  ```

- **Plotting:**

  - Plots actual vs predicted prices and prints R² score.

  ```python
  fig = rf_service.plot_rf_prediction(y_actual_prices, y_pred_prices, date_series)
  ```

- **Improvement:**
  - Predicting price change (delta) improves accuracy for non-stationary financial series and makes the model robust to trend shifts.

---

#### LSTM (Long Short-Term Memory)

- **Data Handling:**

  - Builds sequences of features for each time window (default 30 days), capturing temporal dependencies.
  - Scales features and target separately using `MinMaxScaler`, which is important for neural networks.

  ```python
  X_train, y_train, X_test, y_test, scaler_X, scaler_y, test_dates, original_y_test = LSTMService.process_lstm_data(df, target_col, sequence_length=30, is_training=True)
  ```

- **Training & Saving:**

  - Trains a stacked LSTM with dropout for regularization, then saves the model and scalers.
  - Improved by using dropout and stacked layers, which help prevent overfitting and capture complex patterns.

  ```python
  LSTMService.train_and_save_lstm(X_train, y_train, scaler_X, scaler_y, model_path, scaler_path, sequence_length=30, epochs=50)
  ```

- **Prediction & Inference:**

  - For test prediction, predicts on test sequences and inverse-scales the output.
  - For next-day inference, uses the last sequence to predict the next close price.

  ```python
  y_pred, y_true, dates = LSTMService.load_and_predict_lstm(df, target_col, model_path, scaler_path, sequence_length=30, is_test=True)
  pred_price, pred_date = LSTMService.load_and_predict_lstm(df, target_col, model_path, scaler_path, sequence_length=30, is_test=False)
  ```

- **Plotting:**

  - Plots actual vs predicted prices for the test set.

  ```python
  fig = LSTMService.plot_lstm_results(y_true, y_pred, dates)
  ```

- **Improvement:**
  - Sequence modeling with LSTM captures temporal dependencies and patterns missed by classical models. Dropout and careful scaling further improve generalization.

---

#### ARIMA

- **Data Handling:**

  - Uses only the target close price column, with train/test split by time.
  - Handles missing values by dropping them before modeling.

  ```python
  train_data, test_data, test_dates = ARIMAService.process_data(df, target_code, test_size=0.2)
  ```

- **Training & Saving:**

  - Trains an ARIMA model with specified order and saves it for reuse.
  - Improved by walk-forward validation, which simulates real-world forecasting.

  ```python
  ARIMAService.train_and_save_model(train_data, order=(5,1,0), model_path='arima_model.joblib')
  ```

- **Prediction & Inference:**

  - For test prediction, uses walk-forward validation to predict each test point sequentially.
  - For next-day inference, fits ARIMA to all available data and forecasts one step ahead.

  ```python
  y_pred, y_true, dates = ARIMAService.load_and_predict(df, target_code, model_path='arima_model.joblib', order=(5,1,0), test_size=0.2, is_test=True)
  pred_price, pred_date = ARIMAService.load_and_predict(df, target_code, model_path='arima_model.joblib', order=(5,1,0), test_size=0.2, is_test=False)
  ```

- **Plotting:**

  - Plots actual vs predicted prices with date formatting for clarity.

  ```python
  fig = ARIMAService.plot_prediction(dates, y_true, y_pred, title="ARIMA Actual vs Predicted")
  ```

- **Improvement:**
  - Walk-forward validation provides a realistic estimate of forecasting accuracy and avoids lookahead bias.

---

**Summary of Improvements:**

- All models use time-ordered splits to avoid lookahead bias.
- Feature scaling and saving scalers ensure consistent inference.
- Random Forest and LSTM predict price changes or use sequences, improving accuracy for financial time series.
- Walk-forward validation in ARIMA simulates real-world prediction.
- All models save artifacts for reproducibility and easy deployment.

\*\*These improvements make the modeling pipeline robust, accurate, and suitable for real-time prediction

## 7. Chatbot Implementation

### Library & Architecture

- **Library:** Uses [LangChain](https://python.langchain.com/) and [Google Gemini API](https://ai.google.com/).
- **Integration:** The chatbot is implemented in `src/chatbot.py` and called from `app.py`.

### Context Management

- Each chatbot call includes chart descriptions and context range to minimize token usage and maximize relevance.
- The context is dynamically built from chart annotations and selected data range.

### Reliability

- **Fallbacks:** If the agent fails (e.g., parsing error, API overload), the chatbot falls back to direct Gemini API calls and rotates API keys for reliability.
- **Error Handling:** Handles output parsing errors, model overloads, and iteration/time limits gracefully.
- **Session State:** Chatbot responses and context are managed using Streamlit's `st.session_state` for persistence.

**Code Example:**

```python
# filepath: [app.py]
response = chatbot.ask_langchain_gemini(df, start, end, user_prompt, chart_description=chart_desc)
```

- The chatbot uses the DataFrame and chart description as context.
- If the agent fails, it tries Gemini API directly, rotating keys if needed.

### Notice on Context

- **Context Limitation:** To reduce token usage and improve relevance, only the necessary chart description and data slice are sent to the chatbot.
- **User Experience:** Users can ask questions about any chart, model, or result, making the dashboard interactive and informative.

---

## Getting Started

1. **Install requirements:**  
   `pip install -r requirements.txt`
2. **Set up API keys:**  
   Add your Gemini API keys to `.env`.
3. **Run the app:**  
   `streamlit run src/app.py`
4. **Explore:**  
   Select tickers, time range, and models. Visualize, train, predict, and chat!

---

**For more details, see the code and comments in each service and the main app file.**
