import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os

from config.settings import Config

# Import your existing services
from services.yf_service import YahooFinanceData
from services.news_service import NewsService
from services.macro_service import MacroService
from services.data_prepare_service import DataPrepareService
from services.visualize_service import VisualizeService
from services.linear_regression_service import LinearRegressionService
from services.random_forest_service import RandomForestService
from services.lstm_service import LSTMService
from services.arima_service import ARIMAService

from utils.quick_eda import quick_eda
from chatbot import ChatBot

# --- Chatbot UI Helper ---
def chatbot_ui(df, key_prefix, chatbot, chart_description=None):
    st.markdown("**💬 Message Area**")
    msg_key = f"{key_prefix}_chatbot_msg"
    if msg_key not in st.session_state:
        st.session_state[msg_key] = ""
    if df is not None and not df.empty:
        col1, col2 = st.columns([3, 7])
        with col1:
            context_range = st.slider("Context Range (%)", 0, 100, (0, 100), key=f"{key_prefix}_range")
        with col2:
            user_prompt = st.text_input("Ask a question about this data:", key=f"{key_prefix}_prompt")
        if st.button("Send", key=f"{key_prefix}_send"):
            start, end = context_range
            with st.spinner("Chatbot is thinking..."):
                response = chatbot.ask_langchain_gemini(df, start, end, user_prompt, chart_description=chart_description)
                st.session_state[msg_key] = response
        if st.session_state[msg_key]:
            st.info(st.session_state[msg_key])
    else:
        st.caption("No data available for chatbot analysis.")

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Service Initialization ---
@st.cache_resource
def init_services():
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
    return yf_service, news_service, macro_service, prepare_service, visualize_service, linear_regression_service, random_forest_service, lstm_service, arima_service, chatbot

yf_service, news_service, macro_service, prepare_service, visualize_service, linear_regression_service, random_forest_service, lstm_service, arima_service, chatbot = init_services()

# --- Initialize Session State ---
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.yf_raw_data = None
    st.session_state.news_raw_data = None
    st.session_state.fred_raw_data = None

# --- Callback functions to advance the pipeline ---
def set_step(step_num):
    st.session_state.step = step_num

# --- Main App ---
st.title("Multi-Modal Financial Analysis Dashboard")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Data Selection")
    selected_tickers = st.multiselect("Select Stock Tickers", options=["AAPL", "TSLA", "MSFT", "NVDA"], default=["AAPL", "TSLA"])
    selected_news_queries = st.multiselect("Select News Queries", options=["AAPL", "TSLA", "MSFT", "NVDA", "Stock Market", "AI"], default=["AAPL", "TSLA"])
    selected_macro_indicators = st.multiselect("Select FRED Macro Indicators", options=['FEDFUNDS', 'GDP', 'CPIAUCSL', 'UNRATE'], default=['FEDFUNDS', 'GDP', 'CPIAUCSL'])
    
    st.subheader("Time Range")
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=365))
    end_date = st.date_input("End Date", today)

    # Button to start the entire pipeline
    st.button("🚀 Start Pipeline", on_click=set_step, args=[1], type="primary", use_container_width=True)

    # Add pipeline navigation controls once the pipeline has started
    if st.session_state.step > 0:
        st.markdown("---")
        st.header("🔄 Pipeline Navigation")
        
        st.button("Rerun from Fetching (Step 1)", on_click=set_step, args=[1], use_container_width=True)
        if st.session_state.step > 1:
             st.button("Rerun from Processing (Step 2)", on_click=set_step, args=[2], use_container_width=True)
        st.button("↩️ Reset and Go Home", on_click=set_step, args=[0], use_container_width=True)

# --- Pipeline Header ---
steps = ["Fetch Raw Data", "Process Data", "Visualize Data", "Train Model"]
current_step = st.session_state.step
if current_step > 0:
    cols = st.columns(len(steps))
    for i, col in enumerate(cols):
        with col:
            if i < current_step:
                st.success(f"✅ {steps[i]}")
            elif i == current_step:
                st.info(f"🟦 {steps[i]}")
            else:
                st.write(f"⏭️ {steps[i]}")
    st.markdown("---")

# --- Main Panel for Displaying Pipeline Steps ---

# STEP 1: Fetch Raw Data
if st.session_state.step == 1:
    st.header("Step 1: Fetching Raw Data")
    if not selected_tickers:
        st.warning("Please select at least one stock ticker in the sidebar.")
    else:
        with st.spinner("Fetching data... This may take a moment."):
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            Config.create_directories()

            # Fetch, save, and store data in session state
            st.session_state.yf_raw_data = yf_service.fetch_yahoo_data(selected_tickers, start_date_str, end_date_str)
            st.session_state.yf_raw_data.to_csv(os.path.join(Config.RAW_DATA_DIR, "yf_raw_data.csv"))
            
            st.session_state.news_raw_data = news_service.fetch_news_data(selected_news_queries, start_date_str, end_date_str)
            st.session_state.news_raw_data.to_csv(os.path.join(Config.RAW_DATA_DIR, "news_raw_data.csv"))

            st.session_state.fred_raw_data = macro_service.fetch_fred_data(selected_macro_indicators, start_date_str, end_date_str)
            st.session_state.fred_raw_data.to_csv(os.path.join(Config.RAW_DATA_DIR, "fred_raw_data.csv"))

        st.success(f"Raw data fetched and saved to `{Config.RAW_DATA_DIR}`.")
        with st.expander("View Raw Data Samples"):
            st.subheader("Stock Data")
            stock_desc = (
                f"Raw stock price and volume data fetched from Yahoo Finance for tickers: {', '.join(selected_tickers)} "
                f"from {start_date_str} to {end_date_str}. Each row represents one trading day for each ticker."
            )
            st.caption(stock_desc)
            st.dataframe(st.session_state.yf_raw_data.head())
            st.subheader("News Data")
            news_desc = (
                f"Raw news headlines fetched for queries: {', '.join(selected_news_queries)} "
                f"from {start_date_str} to {end_date_str}. Each row contains news titles for each ticker and date."
            )
            st.caption(news_desc)
            st.dataframe(st.session_state.news_raw_data.head())
            st.subheader("Macro Data")
            macro_desc = (
                f"Raw macroeconomic indicators from FRED for: {', '.join(selected_macro_indicators)} "
                f"from {start_date_str} to {end_date_str}. Each row contains macro values for a given date."
            )
            st.caption(macro_desc)
            st.dataframe(st.session_state.fred_raw_data.head())
        
        st.button("Proceed to Process Data ➡️", on_click=set_step, args=[2])

# STEP 2: Process and Feature Engineer Data
if st.session_state.step == 2:
    st.header("Step 2: Processing and Merging Data")
    with st.spinner("Processing data..."):
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Process data from session state
        featured_stock_data = yf_service.featuring_stock_data(st.session_state.yf_raw_data, start_date_str, end_date_str)
        sentimented_news_data = news_service.sentiment_3_class_news_data(st.session_state.news_raw_data)
        ffilled_macro_data = macro_service.ffill_macro_data(st.session_state.fred_raw_data)
        
        # Merge Data
        final_df = prepare_service.merge_and_fill_nan(featured_stock_data, sentimented_news_data, ffilled_macro_data)
        
        # Save processed data
        processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
        final_df.to_csv(processed_file_path)

        # Store processed data in session state for later steps
        st.session_state.featured_stock_data = featured_stock_data
        st.session_state.sentimented_news_data = sentimented_news_data
        st.session_state.ffilled_macro_data = ffilled_macro_data
        st.session_state.final_df = final_df

    st.success(f"Processed data saved to `{processed_file_path}`.")
    with st.expander("View Processed Data Details"):
        st.subheader("Featured Stock Data")
        featured_desc = (
            "Stock data after feature engineering, including technical indicators (e.g., moving averages, RSI, volatility) "
            "for each ticker and date."
        )
        st.caption(featured_desc)
        st.dataframe(featured_stock_data.head())
        st.subheader("Sentimented News Data")
        sentimented_desc = (
            "News data after sentiment analysis, with sentiment scores (positive, negative, neutral) for each ticker and date."
        )
        st.caption(sentimented_desc)
        st.dataframe(sentimented_news_data.head())
        st.subheader("Forward-Filled Macro Data")
        ffilled_macro_desc = (
            "Macroeconomic data after forward-filling missing values, aligned by date."
        )
        st.caption(ffilled_macro_desc)
        st.dataframe(ffilled_macro_data.head())
        st.subheader("Final Merged DataFrame")
        final_desc = (
            "Final merged DataFrame combining stock features, sentimented news, and macroeconomic data for each date. "
            "This is the main dataset used for visualization and modeling."
        )
        st.caption(final_desc)
        st.dataframe(final_df.head())

    st.button("Proceed to Visualize Data ➡️", on_click=set_step, args=[3])

# STEP 3: Visualize Data
if st.session_state.step == 3:
    st.header("Step 3: Visualizing Data")
    processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
    
    # Load processed data from session state if available, else from disk
    if "final_df" in st.session_state:
        final_df = st.session_state.final_df
    else:
        try:
            final_df = pd.read_csv(processed_file_path)
            st.session_state.final_df = final_df
            st.success("Loaded processed data for visualization.")
        except FileNotFoundError:
            st.error(f"Processed data file not found at `{processed_file_path}`. Please go back to Step 2 to generate it.")
            st.stop()

    # --- General Analysis ---
    st.subheader("Overall Data Analysis")
    with st.expander("Exploratory Data Analysis (EDA)"):
        eda_df = final_df.copy()
        if "eda_df" not in st.session_state:
            st.session_state.eda_df = eda_df
        eda_desc = (
            "Exploratory Data Analysis (EDA) on the final merged dataset. "
            "This includes summary statistics, distributions, and basic insights."
        )
        st.caption(eda_desc)
        quick_eda(st.session_state.eda_df)

    with st.expander("Correlation Heatmap"):
        if "corr_fig" not in st.session_state or "corr_df" not in st.session_state:
            corr_fig, corr_df = visualize_service.create_correlation_heatmap(final_df)
            st.session_state.corr_fig = corr_fig
            st.session_state.corr_df = corr_df
        corr_desc = (
            "Correlation heatmap showing the pairwise correlation coefficients between all numerical features in the final merged dataset. "
            "This helps identify relationships between features."
        )
        if st.session_state.corr_fig:
            st.caption(corr_desc)
            st.plotly_chart(st.session_state.corr_fig, use_container_width=True)
            chatbot_ui(st.session_state.corr_df, "corr_heatmap", chatbot, corr_desc)
    
    # --- Ticker-Specific Analysis ---
    st.subheader("Ticker-Specific Analysis")
    for ticker in selected_tickers:
        st.markdown("---")
        st.header(f"Visuals for {ticker}")
        
        # Price and Volume
        wc_key = f"wordcloud_{ticker}"
        ohlcv_key = f"ohlcv_{ticker}"
        returns_key = f"returns_{ticker}"
        sentiment_key = f"sentiment_{ticker}"

        if ohlcv_key not in st.session_state:
            ohlcv_fig, ohlcv_df = visualize_service.create_ohlcv_fig(final_df, ticker)
            st.session_state[ohlcv_key] = (ohlcv_fig, ohlcv_df)
        ohlcv_fig, ohlcv_df = st.session_state[ohlcv_key]
        ohlcv_desc = (
            f"OHLCV (Open, High, Low, Close, Volume) candlestick chart for {ticker}. "
            "Shows daily price movement and trading volume for the selected period."
        )
        if ohlcv_fig:
            st.caption(ohlcv_desc)
            st.plotly_chart(ohlcv_fig, use_container_width=True)
            chatbot_ui(ohlcv_df, ohlcv_key, chatbot, ohlcv_desc)
        
        # Daily Returns
        if returns_key not in st.session_state:
            returns_fig, returns_df = visualize_service.create_daily_return_histogram(final_df, ticker)
            st.session_state[returns_key] = (returns_fig, returns_df)
        returns_fig, returns_df = st.session_state[returns_key]
        returns_desc = (
            f"Histogram of daily returns for {ticker}. "
            "Shows the distribution of percentage changes in closing price from one day to the next."
        )
        if returns_fig:
            st.caption(returns_desc)
            st.plotly_chart(returns_fig, use_container_width=True)
            chatbot_ui(returns_df, returns_key, chatbot, returns_desc)

        # Sentiment Analysis
        if sentiment_key not in st.session_state:
            sentiment_fig, sentiment_df = visualize_service.create_sentiment_line_chart(final_df, ticker)
            st.session_state[sentiment_key] = (sentiment_fig, sentiment_df)
        sentiment_fig, sentiment_df = st.session_state[sentiment_key]
        sentiment_desc = (
            f"Line chart of sentiment scores (positive, negative, neutral) for {ticker} over time, "
            "derived from news headlines."
        )
        if sentiment_fig:
            st.caption(sentiment_desc)
            st.plotly_chart(sentiment_fig, use_container_width=True)
            chatbot_ui(sentiment_df, sentiment_key, chatbot, sentiment_desc)

        # Word Cloud from Raw News Data
        if st.session_state.news_raw_data is not None:
            if wc_key not in st.session_state:
                wordcloud_fig, wordcloud_df = visualize_service.create_news_wordcloud_figure(st.session_state.news_raw_data, ticker)
                st.session_state[wc_key] = (wordcloud_fig, wordcloud_df)
            wordcloud_fig, wordcloud_df = st.session_state[wc_key]
            wordcloud_desc = (
                f"Word cloud of news headlines for {ticker}. "
                "Shows the most frequent words in news titles for the selected period."
            )
            if wordcloud_fig:
                st.caption(wordcloud_desc)
                st.pyplot(wordcloud_fig)
                chatbot_ui(wordcloud_df, wc_key, chatbot, wordcloud_desc)
        else:
            st.warning(f"Raw news data for {ticker} not available in session state for word cloud.")

    # --- Macroeconomic Analysis ---
    st.subheader("Macroeconomic Analysis")
    with st.expander("View Macroeconomic Indicator Charts"):
        for indicator in selected_macro_indicators:
            macro_key = f"macro_{indicator}"
            if macro_key not in st.session_state:
                macro_fig, macro_df = visualize_service.create_macro_timeseries_line_chart(final_df, indicator)
                st.session_state[macro_key] = (macro_fig, macro_df)
            macro_fig, macro_df = st.session_state[macro_key]
            macro_desc = (
                f"Line chart of the macroeconomic indicator '{indicator}' over time. "
                "Shows how this macro variable changes during the selected period."
            )
            if macro_fig:
                st.caption(macro_desc)
                st.plotly_chart(macro_fig, use_container_width=True)
                chatbot_ui(macro_df, macro_key, chatbot, macro_desc)

    # --- Technical Analysis (All Tickers) ---
    st.subheader("Technical Indicator Analysis")
    with st.expander("RSI Charts"):
        if "rsi_figs" not in st.session_state:
            st.session_state.rsi_figs = visualize_service.create_rsi_figs(final_df)
        rsi_figs = st.session_state.rsi_figs
        rsi_desc = (
            f"Relative Strength Index (RSI) chart for each ticker. "
            "Shows RSI14 and RSI21 values over time, indicating overbought/oversold conditions."
        )
        if rsi_figs:
            for idx, (_, fig, rsi_df) in enumerate(rsi_figs):
                st.caption(rsi_desc)
                st.plotly_chart(fig, use_container_width=True)
                chatbot_ui(rsi_df, f"rsi_{idx}", chatbot, rsi_desc)
        else:
            st.write("No RSI data to display.")
            
    with st.expander("Moving Average Charts"):
        if "ma_figs" not in st.session_state:
            st.session_state.ma_figs = visualize_service.create_ma_figs(final_df)
        ma_figs = st.session_state.ma_figs
        ma_desc = (
            f"Moving Average (MA) chart for each ticker. "
            "Shows MA50 and MA200 values over time, indicating trend direction."
        )
        if ma_figs:
            for idx, (_, fig, ma_df) in enumerate(ma_figs):
                st.caption(ma_desc)
                st.plotly_chart(fig, use_container_width=True)
                chatbot_ui(ma_df, f"ma_{idx}", chatbot, ma_desc)
        else:
            st.write("No Moving Average data to display.")
        
    # --- Missing Value Bar Chart ---
    st.subheader("Missing Value Analysis")
    with st.expander("Missing Value Bar Chart"):
        if "missing_fig" not in st.session_state or "missing_df" not in st.session_state:
            missing_fig, missing_df = visualize_service.create_missing_value_bar_chart(final_df)
            st.session_state.missing_fig = missing_fig
            st.session_state.missing_df = missing_df
        missing_desc = (
            "Bar chart showing the count and percentage of missing values for each column in the final merged dataset."
        )
        if st.session_state.missing_fig:
            st.caption(missing_desc)
            st.plotly_chart(st.session_state.missing_fig, use_container_width=True)
            chatbot_ui(st.session_state.missing_df, "missing_values", chatbot, missing_desc)
        else:
            st.write("No missing values detected.")

    st.button("Proceed to Train Model ➡️", on_click=set_step, args=[4])

# STEP 4: Model Training
if st.session_state.step == 4:
    st.header("Step 4: Model Training & Prediction")
    
    # Select target ticker for all models
    target_ticker = st.selectbox(
        "Select Target Ticker for Prediction",
        options=selected_tickers,
        index=0,
        key='model_target_ticker'
    )
    
    st.markdown("---")

    # --- Linear Regression Section ---
    with st.expander("Linear Regression Model"):
        if st.button(f"Train and Evaluate LR for {target_ticker}", key="train_lr_button"):
            processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
            try:
                final_df = pd.read_csv(processed_file_path)
            except FileNotFoundError:
                st.error(f"Processed data file not found. Please go back to Step 2 to generate it.")
                st.stop()

            with st.spinner(f"Training Linear Regression model for {target_ticker}..."):
                model_path = os.path.join(Config.MODELS_DIR, f"lr_model_{target_ticker}.joblib")
                scaler_path = os.path.join(Config.MODELS_DIR, f"lr_scaler_{target_ticker}.joblib")

                X_tr, X_te, y_tr, y_te, scaler, dates_te = linear_regression_service.process_data(
                    final_df, target_ticker, is_training=True
                )
                st.write("✅ Data processed and split for training/testing.")

                linear_regression_service.train_and_save_model(X_tr, y_tr, scaler, model_path, scaler_path)
                st.write(f"✅ Model and scaler for {target_ticker} trained and saved.")

                y_pred, y_true, dates = linear_regression_service.load_and_predict(
                    final_df, target_ticker, model_path, scaler_path, is_test=True
                )
                st.write("✅ Model evaluated on the test set.")

                fig = linear_regression_service.create_actual_vs_predicted_figure(y_true, y_pred, dates, title=f"Linear Regression: Actual vs. Predicted for {target_ticker}")
                lr_desc = (
                    f"Actual vs. Predicted closing prices for {target_ticker} using a Linear Regression model. "
                    "The plot compares model predictions to true values on the test set."
                )
                st.caption(lr_desc)
                st.session_state.lr_fig = fig
                st.pyplot(fig)

                lr_results_df = pd.DataFrame({
                    "date": dates,
                    "actual": y_true,
                    "predicted": y_pred
                })
                st.session_state.lr_results_df = lr_results_df
                chatbot_ui(lr_results_df, f"lr_{target_ticker}", chatbot, lr_desc)

                pred_price, pred_date = linear_regression_service.load_and_predict(
                    final_df, target_ticker, model_path, scaler_path, is_test=False
                )
                st.session_state.lr_pred_price = pred_price
                st.session_state.lr_pred_date = pred_date
                st.success(f"Predicted Next-Day Close Price for {target_ticker} on {pd.to_datetime(pred_date).date() + timedelta(days=1)}: **${pred_price:.2f}**")

        # Show stored results if available
        if "lr_fig" in st.session_state:
            st.pyplot(st.session_state.lr_fig)
        if "lr_results_df" in st.session_state:
            chatbot_ui(st.session_state.lr_results_df, f"lr_{target_ticker}", chatbot, "Linear Regression Results")
        if "lr_pred_price" in st.session_state and "lr_pred_date" in st.session_state:
            st.success(f"Predicted Next-Day Close Price for {target_ticker} on {pd.to_datetime(st.session_state.lr_pred_date).date() + timedelta(days=1)}: **${st.session_state.lr_pred_price:.2f}**")

    # --- Random Forest Section ---
    with st.expander("Random Forest Model (Predicts Price Change)"):
        if st.button(f"Train and Evaluate RF for {target_ticker}", key="train_rf_button"):
            processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
            try:
                final_df = pd.read_csv(processed_file_path)
            except FileNotFoundError:
                st.error(f"Processed data file not found. Please go back to Step 2 to generate it.")
                st.stop()

            with st.spinner(f"Training Random Forest model for {target_ticker}..."):
                rf_model_path = os.path.join(Config.MODELS_DIR, f"rf_model_{target_ticker}.joblib")
                rf_scaler_path = os.path.join(Config.MODELS_DIR, f"rf_scaler_{target_ticker}.joblib")
                
                target_col = f"{target_ticker}_close_stock"
                features_to_drop = ['Date', 'target_diff'] + [col for col in final_df.columns if col.endswith('_close_stock')]
                feature_names = [col for col in final_df.columns if col not in features_to_drop]

                X_train, _, y_train, _, scaler, _, _ = random_forest_service.process_rf_features(final_df, target_ticker, is_training=True)
                st.write("✅ Data processed for Random Forest.")

                random_forest_service.train_and_save_rf_model(X_train, y_train, scaler, rf_model_path, rf_scaler_path)
                st.write(f"✅ Random Forest model and scaler for {target_ticker} trained and saved.")

                y_pred_prices, y_actual_prices, dates = random_forest_service.load_rf_model_and_predict(final_df, target_ticker, rf_model_path, rf_scaler_path, is_test=True)
                fig_rf = random_forest_service.plot_rf_prediction(y_actual_prices, y_pred_prices, dates, title=f"RF: Actual vs. Predicted for {target_ticker}")
                rf_desc = (
                    f"Actual vs. Predicted closing prices for {target_ticker} using a Random Forest model. "
                    "The plot compares model predictions to true values on the test set."
                )
                st.caption(rf_desc)
                st.session_state.rf_fig = fig_rf
                st.plotly_chart(fig_rf)

                rf_results_df = pd.DataFrame({
                    "date": dates,
                    "actual": y_actual_prices,
                    "predicted": y_pred_prices
                })
                st.session_state.rf_results_df = rf_results_df
                chatbot_ui(rf_results_df, f"rf_{target_ticker}", chatbot, rf_desc)

                pred_price_rf = random_forest_service.load_rf_model_and_predict(final_df, target_ticker, rf_model_path, rf_scaler_path, is_test=False)
                last_known_date = pd.to_datetime(final_df['Date']).iloc[-1]
                st.session_state.rf_pred_price = pred_price_rf
                st.session_state.rf_pred_date = last_known_date
                st.success(f"RF Predicted Next-Day Close for {target_ticker} on {last_known_date.date() + timedelta(days=1)}: **${pred_price_rf:.2f}**")

        # Show stored results if available
        if "rf_fig" in st.session_state:
            st.plotly_chart(st.session_state.rf_fig)
        if "rf_results_df" in st.session_state:
            chatbot_ui(st.session_state.rf_results_df, f"rf_{target_ticker}", chatbot, "Random Forest Results")
        if "rf_pred_price" in st.session_state and "rf_pred_date" in st.session_state:
            st.success(f"RF Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(st.session_state.rf_pred_date).date() + timedelta(days=1)}: **${st.session_state.rf_pred_price:.2f}**")

    # --- LSTM Section ---
    with st.expander("LSTM Model"):
        seq_length = st.slider("Select Sequence Length (days)", min_value=10, max_value=90, value=30, key="lstm_seq")
        if st.button(f"Train and Evaluate LSTM for {target_ticker}", key="train_lstm_button"):
            processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
            try:
                final_df = pd.read_csv(processed_file_path)
            except FileNotFoundError:
                st.error(f"Processed data file not found. Please go back to Step 2 to generate it.")
                st.stop()

            with st.spinner(f"Training LSTM model for {target_ticker} (this may take a while)..."):
                lstm_model_path = os.path.join(Config.MODELS_DIR, f"lstm_model_{target_ticker}.h5")
                lstm_scaler_path = os.path.join(Config.MODELS_DIR, f"lstm_scalers_{target_ticker}.joblib")
                target_col_lstm = f"{target_ticker}_close_stock"

                X_train, y_train, _, _, scaler_X, scaler_y, _, _ = lstm_service.process_lstm_data(
                    final_df, target_col_lstm, sequence_length=seq_length, is_training=True
                )
                st.write("✅ Data processed for LSTM.")

                lstm_service.train_and_save_lstm(
                    X_train, y_train, scaler_X, scaler_y, lstm_model_path, lstm_scaler_path, sequence_length=seq_length
                )
                st.write(f"✅ LSTM model and scalers for {target_ticker} trained and saved.")

                y_pred, y_true, dates = lstm_service.load_and_predict_lstm(
                    final_df, target_col_lstm, lstm_model_path, lstm_scaler_path, sequence_length=seq_length, is_test=True
                )
                fig_lstm = lstm_service.plot_lstm_results(y_true, y_pred, dates, title=f"LSTM: Actual vs. Predicted for {target_ticker}")
                lstm_desc = (
                    f"Actual vs. Predicted closing prices for {target_ticker} using an LSTM model. "
                    "The plot compares model predictions to true values on the test set."
                )
                st.caption(lstm_desc)
                st.session_state.lstm_fig = fig_lstm
                st.pyplot(fig_lstm)

                lstm_results_df = pd.DataFrame({
                    "date": dates,
                    "actual": y_true,
                    "predicted": y_pred
                })
                st.session_state.lstm_results_df = lstm_results_df
                chatbot_ui(lstm_results_df, f"lstm_{target_ticker}", chatbot, lstm_desc)

                pred_price_lstm, pred_date_lstm = lstm_service.load_and_predict_lstm(
                    final_df, target_col_lstm, lstm_model_path, lstm_scaler_path, sequence_length=seq_length, is_test=False
                )
                st.session_state.lstm_pred_price = pred_price_lstm
                st.session_state.lstm_pred_date = pred_date_lstm
                st.success(f"LSTM Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(pred_date_lstm).date() + timedelta(days=1)}: **${pred_price_lstm:.2f}**")

        # Show stored results if available
        if "lstm_fig" in st.session_state:
            st.pyplot(st.session_state.lstm_fig)
        if "lstm_results_df" in st.session_state:
            chatbot_ui(st.session_state.lstm_results_df, f"lstm_{target_ticker}", chatbot, "LSTM Results")
        if "lstm_pred_price" in st.session_state and "lstm_pred_date" in st.session_state:
            st.success(f"LSTM Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(st.session_state.lstm_pred_date).date() + timedelta(days=1)}: **${st.session_state.lstm_pred_price:.2f}**")

    # --- ARIMA Section ---
    with st.expander("ARIMA Model (Time Series Forecast)"):
        if st.button(f"Train and Forecast ARIMA for {target_ticker}", key="train_arima_button"):
            processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
            try:
                final_df = pd.read_csv(processed_file_path)
            except FileNotFoundError:
                st.error(f"Processed data file not found. Please go back to Step 2 to generate it.")
                st.stop()

            with st.spinner(f"Training ARIMA model for {target_ticker}..."):
                model_path = os.path.join(Config.MODELS_DIR, f"arima_model_{target_ticker}.joblib")
                # Test set prediction
                y_pred, y_true, dates = arima_service.load_and_predict(final_df, target_ticker, model_path=model_path, order=(5,1,0), test_size=0.2, is_test=True)
                arima_desc = (
                    f"ARIMA time series forecast for {target_ticker} closing price. "
                    "Shows actual close prices and ARIMA prediction for the test set."
                )
                st.caption(arima_desc)
                fig = arima_service.plot_prediction(dates, y_true, y_pred, title=f"ARIMA: Actual vs. Predicted for {target_ticker}")
                st.session_state.arima_fig = fig
                st.pyplot(fig)

                arima_results_df = pd.DataFrame({
                    "date": dates,
                    "actual": y_true,
                    "predicted": y_pred
                })
                st.session_state.arima_results_df = arima_results_df
                chatbot_ui(arima_results_df, f"arima_{target_ticker}", chatbot, arima_desc)

                # Inference: predict next day's close
                pred_price, pred_date = arima_service.load_and_predict(final_df, target_ticker, model_path=model_path, order=(5,1,0), test_size=0.2, is_test=False)
                st.session_state.arima_pred_price = pred_price
                st.session_state.arima_pred_date = pred_date
                st.success(f"ARIMA Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(pred_date).date()}: **${pred_price:.2f}**")

        # Show stored results if available
        if "arima_fig" in st.session_state:
            st.pyplot(st.session_state.arima_fig)
        if "arima_results_df" in st.session_state:
            chatbot_ui(st.session_state.arima_results_df, f"arima_{target_ticker}", chatbot, "ARIMA Results")
        if "arima_pred_price" in st.session_state and "arima_pred_date" in st.session_state:
            st.success(f"ARIMA Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(st.session_state.arima_pred_date).date()}: **${st.session_state.arima_pred_price:.2f}**")

    # --- TGNN Section ---
    with st.expander("TGNN++ Model (Graph Transformer, Pretrained Only)"):
        st.info(
            "⚠️ TGNN++ is a complex deep learning model and training is disabled for performance reasons. "
            "You can only use the pretrained model (`tgnnpp_finbert_model_best.pth`) for test set evaluation and next-day inference."
        )
        processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
        tgnn_model_path = os.path.join(Config.MODELS_DIR, "tgnnpp_finbert_model_best.pth")
        try:
            final_df = pd.read_csv(processed_file_path)
        except FileNotFoundError:
            st.error(f"Processed data file not found. Please go back to Step 2 to generate it.")
            st.stop()

        # Prepare TGNN data (same as training, but only for prediction)
        from services.tgnn_service import prepare_tgnn_data, load_tgnn_model, predict_tgnn

        with st.spinner("Preparing TGNN data..."):
            dataset, stock_codes, feature_dim, split_idx = prepare_tgnn_data(final_df, target_stock=target_ticker, embedding_dim=3)

        # Load pretrained TGNN model
        with st.spinner("Loading pretrained TGNN++ model..."):
            model, device = load_tgnn_model(num_stocks=len(stock_codes), feature_dim=feature_dim, window_size=30, model_path=tgnn_model_path)

        # Test set prediction
        if "tgnn_test_results" not in st.session_state or "tgnn_test_fig" not in st.session_state:
            test_preds, test_targets = predict_tgnn(dataset, model, device, split_idx, mode="test")
            tgnn_test_df = pd.DataFrame({
                "actual": test_targets,
                "predicted": test_preds
            })
            st.session_state.tgnn_test_results = tgnn_test_df

            # Plot actual vs predicted
            import plotly.graph_objs as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=test_targets, mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(y=test_preds, mode='lines+markers', name='Predicted'))
            fig.update_layout(
                title=f"TGNN++: Actual vs. Predicted Close Price for {target_ticker} (Test Set)",
                xaxis_title="Test Time Step",
                yaxis_title="Close Price",
                template="plotly_white",
                height=400
            )
            st.session_state.tgnn_test_fig = fig

        tgnn_desc = (
            f"Actual vs. Predicted closing prices for {target_ticker} using the pretrained TGNN++ model. "
            "The plot compares model predictions to true values on the test set."
        )
        st.caption(tgnn_desc)
        st.plotly_chart(st.session_state.tgnn_test_fig, use_container_width=True)
        chatbot_ui(st.session_state.tgnn_test_results, f"tgnn_{target_ticker}_test", chatbot, tgnn_desc)

        # Next-day inference
        if "tgnn_pred_price" not in st.session_state:
            pred_price = predict_tgnn(dataset, model, device, split_idx, mode="inference")
            st.session_state.tgnn_pred_price = pred_price
            last_date = pd.to_datetime(final_df['Date']).iloc[-1]
            st.session_state.tgnn_pred_date = last_date

        st.success(
            f"TGNN++ Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(st.session_state.tgnn_pred_date).date() + timedelta(days=1)}: "
            f"**${st.session_state.tgnn_pred_price:.2f}**"
        )

    st.button("↩️ Restart Pipeline", on_click=set_step, args=[0])

# --- Home / About Page ---
if st.session_state.step == 0:
    with open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding="utf-8") as f:
        readme_content = f.read()
    st.markdown(readme_content, unsafe_allow_html=True)
    st.info("Use the sidebar to select stock tickers, news queries, and macroeconomic indicators. "
             "Then, navigate through the pipeline steps to fetch, process, visualize data, and train predictive models.")
