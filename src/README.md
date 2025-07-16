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
# filepath: [app.py](http://_vscodecontentref_/0)
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
- **News Data:** Headlines and articles for tickers and market topics via `news_service.py`.
- **Macroeconomic Data:** Indicators (e.g., GDP, CPI, FEDFUNDS) from FRED via `macro_service.py`.

### Data Handling

- **Raw Data:** Saved in CSV files for reproducibility (`Config.RAW_DATA_DIR`).
- **Cleaning:** Remove missing or invalid entries, align by date/ticker.
- **Alignment:** Merge datasets by date and ticker using `prepare_service.merge_and_fill_nan`.
- **Sentiment Analysis:** News headlines are scored for positive, negative, neutral sentiment using NLP models.

**Code Example:**

```python
# filepath: [app.py](http://_vscodecontentref_/2)
st.session_state.yf_raw_data = yf_service.fetch_yahoo_data(selected_tickers, start_date_str, end_date_str)
st.session_state.news_raw_data = news_service.fetch_news_data(selected_news_queries, start_date_str, end_date_str)
st.session_state.fred_raw_data = macro_service.fetch_fred_data(selected_macro_indicators, start_date_str, end_date_str)
```

### Data Meaning

- **Stock Data:** Daily OHLCV (Open, High, Low, Close, Volume) for each ticker.
- **News Data:** Textual context for market events and sentiment.
- **Macro Data:** Economic context for broader market trends.

---

## 4. Feature Engineering & Missing Value Handling

### Feature Engineering

- **Technical Indicators:** Moving averages (MA50, MA200), RSI (14, 21), volatility, lag features via `yf_service.featuring_stock_data`.
- **Returns:** Daily and log returns.
- **Sentiment Features:** Aggregated news sentiment scores via `news_service.sentiment_3_class_news_data`.

### Missing Value Filling

- **Forward Fill:** Macro data is forward-filled using `macro_service.ffill_macro_data` to align with stock dates.
- **Imputation:** For features, missing values are filled using statistical techniques (mean, median, or forward fill) in `prepare_service.merge_and_fill_nan`.

**Purpose:**

- Enhance predictive power.
- Provide richer context for models.
- Ensure no gaps in time series for modeling.

**Code Example:**

```python
featured_stock_data = yf_service.featuring_stock_data(...)
sentimented_news_data = news_service.sentiment_3_class_news_data(...)
ffilled_macro_data = macro_service.ffill_macro_data(...)
final_df = prepare_service.merge_and_fill_nan(featured_stock_data, sentimented_news_data, ffilled_macro_data)
```

---

## 5. Data Visualization

### Need & Reasoning

- **Understanding:** Visualizations help users grasp data distributions, trends, and relationships.
- **Communication:** Charts make results accessible and interpretable for all users.

### Purpose & Usage of Each Chart

- **EDA Summary:** Quick stats and distributions for overall understanding (`quick_eda(final_df)`).
- **Correlation Heatmap:** Identify relationships between features (`visualize_service.create_correlation_heatmap(final_df)`).
- **OHLCV Chart:** Visualize price and volume movements (`visualize_service.create_ohlcv_fig(final_df, ticker)`).
- **Returns Histogram:** Show volatility and risk (`visualize_service.create_daily_return_histogram(final_df, ticker)`).
- **Sentiment Line Chart:** Track market mood over time (`visualize_service.create_sentiment_line_chart(final_df, ticker)`).
- **Word Cloud:** Highlight frequent news topics (`visualize_service.create_news_wordcloud_figure(news_raw_data, ticker)`).
- **Macro Timeseries:** Show economic trends (`visualize_service.create_macro_timeseries_line_chart(final_df, indicator)`).
- **RSI & MA Charts:** Reveal technical signals (`visualize_service.create_rsi_figs(final_df)`, `visualize_service.create_ma_figs(final_df)`).
- **Missing Value Bar Chart:** Diagnose data quality (`visualize_service.create_missing_value_bar_chart(final_df)`).

**Code Example:**

```python
# filepath: [app.py](http://_vscodecontentref_/4)
corr_fig, corr_df = visualize_service.create_correlation_heatmap(final_df)
st.plotly_chart(corr_fig)
chatbot_ui(corr_df, "corr_heatmap", chatbot, corr_desc)

ohlcv_fig, ohlcv_df = visualize_service.create_ohlcv_fig(final_df, ticker)
st.plotly_chart(ohlcv_fig)
chatbot_ui(ohlcv_df, f"ohlcv_{ticker}", chatbot, ohlcv_desc)
```

---

## 6. Model Services

### What Are Those

- **Linear Regression:** Simple baseline for price prediction (`linear_regression_service.py`).
- **Random Forest:** Ensemble method for robust, non-linear predictions (`random_forest_service.py`).
- **LSTM:** Deep learning for sequential/time series modeling (`lstm_service.py`).
- **ARIMA:** Classical time series model for forecasting (`arima_service.py`).

### Purpose

- Compare different modeling approaches.
- Demonstrate strengths/weaknesses of each method.
- Enable hands-on experimentation.

### How They Are Trained

- **process_data:** Prepare features and targets for training.
- **train_and_save_model:** Train the model and save it for reuse.
- **load_and_predict:** Predict on test set (for evaluation) or next day (for inference).
- **plot_prediction:** Visualize actual vs predicted values.

**Code Example:**

```python
# filepath: [app.py](http://_vscodecontentref_/5)
# Linear Regression
X_tr, X_te, y_tr, y_te, scaler, dates_te = linear_regression_service.process_data(final_df, target_ticker, is_training=True)
linear_regression_service.train_and_save_model(X_tr, y_tr, scaler, model_path, scaler_path)
y_pred, y_true, dates = linear_regression_service.load_and_predict(final_df, target_ticker, model_path, scaler_path, is_test=True)
fig = linear_regression_service.create_actual_vs_predicted_figure(y_true, y_pred, dates, ...)
st.pyplot(fig)
pred_price, pred_date = linear_regression_service.load_and_predict(final_df, target_ticker, model_path, scaler_path, is_test=False)
st.success(f"Predicted Next-Day Close Price: ${pred_price:.2f}")

# ARIMA
y_pred, y_true, dates = arima_service.load_and_predict(final_df, target_ticker, model_path=model_path, order=(5,1,0), test_size=0.2, is_test=True)
fig = arima_service.plot_prediction(dates, y_true, y_pred, title=f"ARIMA: Actual vs. Predicted for {target_ticker}")
st.pyplot(fig)
pred_price, pred_date = arima_service.load_and_predict(final_df, target_ticker, model_path=model_path, order=(5,1,0), test_size=0.2, is_test=False)
st.success(f"ARIMA Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(pred_date).date()}: **${pred_price:.2f}**")
```

### How They Predict Price

- Models use processed features to predict closing prices for selected tickers.
- Predictions are shown for both historical test data and future (next day) inference.

---

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
# filepath: [app.py](http://_vscodecontentref_/6)
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
