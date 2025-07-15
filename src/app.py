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
from services.lstm_service import LSTMService # Import the new service
from utils.quick_eda import quick_eda
from chatbot import ChatBot

# --- Chatbot UI Helper ---
def chatbot_ui(df, key_prefix, chatbot):
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
                response = chatbot.ask_langchain_gemini(df, start, end, user_prompt)
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
    chatbot = ChatBot()
    return yf_service, news_service, macro_service, prepare_service, visualize_service, linear_regression_service, random_forest_service, lstm_service, chatbot

yf_service, news_service, macro_service, prepare_service, visualize_service, linear_regression_service, random_forest_service, lstm_service, chatbot = init_services()

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
    selected_news_queries = st.multiselect("Select News Queries", options=["AAPL", "TSLA", "Stock Market", "AI"], default=["AAPL", "TSLA"])
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
        
        # Button to restart the whole pipeline from the beginning
        st.button("Rerun from Fetching (Step 1)", on_click=set_step, args=[1], use_container_width=True)

        # Button to restart from the processing step
        # This is only enabled if the user is past step 1
        if st.session_state.step > 1:
             st.button("Rerun from Processing (Step 2)", on_click=set_step, args=[2], use_container_width=True)
        
        # Button to reset the entire app to its initial state
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
            st.dataframe(st.session_state.yf_raw_data.head())
            st.subheader("News Data")
            st.dataframe(st.session_state.news_raw_data.head())
            st.subheader("Macro Data")
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

    st.success(f"Processed data saved to `{processed_file_path}`.")
    with st.expander("View Processed Data Details"):
        st.subheader("Featured Stock Data")
        st.dataframe(featured_stock_data.head())
        st.subheader("Sentimented News Data")
        st.dataframe(sentimented_news_data.head())
        st.subheader("Forward-Filled Macro Data")
        st.dataframe(ffilled_macro_data.head())
        st.subheader("Final Merged DataFrame")
        st.dataframe(final_df.head())

    st.button("Proceed to Visualize Data ➡️", on_click=set_step, args=[3])

# STEP 3: Visualize Data
if st.session_state.step == 3:
    st.header("Step 3: Visualizing Data")
    processed_file_path = os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv")
    
    try:
        final_df = pd.read_csv(processed_file_path)
        st.success("Loaded processed data for visualization.")
    except FileNotFoundError:
        st.error(f"Processed data file not found at `{processed_file_path}`. Please go back to Step 2 to generate it.")
        st.stop()

    # --- General Analysis ---
    st.subheader("Overall Data Analysis")
    with st.expander("Exploratory Data Analysis (EDA)"):
        eda_df = pd.read_csv(processed_file_path, index_col='Date', parse_dates=True)
        quick_eda(eda_df)

    with st.expander("Correlation Heatmap"):
        corr_fig, corr_df = visualize_service.create_correlation_heatmap(final_df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
            chatbot_ui(corr_df, "corr_heatmap", chatbot)
    
    # --- Ticker-Specific Analysis ---
    st.subheader("Ticker-Specific Analysis")
    for ticker in selected_tickers:
        st.markdown(f"---")
        st.header(f"Visuals for {ticker}")
        
        # Price and Volume
        ohlcv_fig, ohlcv_df = visualize_service.create_ohlcv_fig(final_df, ticker)
        if ohlcv_fig:
            st.plotly_chart(ohlcv_fig, use_container_width=True)
            chatbot_ui(ohlcv_df, f"ohlcv_{ticker}", chatbot)
        
        # Daily Returns
        returns_fig, returns_df = visualize_service.create_daily_return_histogram(final_df, ticker)
        if returns_fig:
            st.plotly_chart(returns_fig, use_container_width=True)
            chatbot_ui(returns_df, f"returns_{ticker}", chatbot)

        # Sentiment Analysis
        sentiment_fig, sentiment_df = visualize_service.create_sentiment_line_chart(final_df, ticker)
        if sentiment_fig:
            st.plotly_chart(sentiment_fig, use_container_width=True)
            chatbot_ui(sentiment_df, f"sentiment_{ticker}", chatbot)

        # Word Cloud from Raw News Data
        if st.session_state.news_raw_data is not None:
            wordcloud_fig, wordcloud_df = visualize_service.create_news_wordcloud_figure(st.session_state.news_raw_data, ticker)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
                chatbot_ui(wordcloud_df, f"wordcloud_{ticker}", chatbot)
        else:
            st.warning(f"Raw news data for {ticker} not available in session state for word cloud.")

    # --- Macroeconomic Analysis ---
    st.subheader("Macroeconomic Analysis")
    with st.expander("View Macroeconomic Indicator Charts"):
        for indicator in selected_macro_indicators:
            macro_fig, macro_df = visualize_service.create_macro_timeseries_line_chart(final_df, indicator)
            if macro_fig:
                st.plotly_chart(macro_fig, use_container_width=True)
                chatbot_ui(macro_df, f"macro_{indicator}", chatbot)

    # --- Technical Analysis (All Tickers) ---
    st.subheader("Technical Indicator Analysis")
    with st.expander("RSI Charts"):
        rsi_figs = visualize_service.create_rsi_figs(final_df)
        if rsi_figs:
            for idx, (_, fig, rsi_df) in enumerate(rsi_figs):
                st.plotly_chart(fig, use_container_width=True)
                chatbot_ui(rsi_df, f"rsi_{idx}", chatbot)
        else:
            st.write("No RSI data to display.")
            
    with st.expander("Moving Average Charts"):
        ma_figs = visualize_service.create_ma_figs(final_df)
        if ma_figs:
            for idx, (_, fig, ma_df) in enumerate(ma_figs):
                st.plotly_chart(fig, use_container_width=True)
                chatbot_ui(ma_df, f"ma_{idx}", chatbot)
        else:
            st.write("No Moving Average data to display.")
        
    # --- Missing Value Bar Chart ---
    st.subheader("Missing Value Analysis")
    with st.expander("Missing Value Bar Chart"):
        missing_fig, missing_df = visualize_service.create_missing_value_bar_chart(final_df)
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True)
            chatbot_ui(missing_df, "missing_values", chatbot)
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
                # Define model and scaler paths
                model_path = os.path.join(Config.MODELS_DIR, f"lr_model_{target_ticker}.joblib")
                scaler_path = os.path.join(Config.MODELS_DIR, f"lr_scaler_{target_ticker}.joblib")

                # 1. Process Data
                X_tr, X_te, y_tr, y_te, scaler, dates_te = linear_regression_service.process_data(
                    final_df, target_ticker, is_training=True
                )
                st.write("✅ Data processed and split for training/testing.")

                # 2. Train and Save Model
                linear_regression_service.train_and_save_model(X_tr, y_tr, scaler, model_path, scaler_path)
                st.write(f"✅ Model and scaler for {target_ticker} trained and saved.")

                # 3. Evaluate on Test Set
                y_pred, y_true, dates = linear_regression_service.load_and_predict(
                    final_df, target_ticker, model_path, scaler_path, is_test=True
                )
                st.write("✅ Model evaluated on the test set.")

                # 4. Plot Results
                fig = linear_regression_service.create_actual_vs_predicted_figure(y_true, y_pred, dates, title=f"Linear Regression: Actual vs. Predicted for {target_ticker}")
                st.pyplot(fig)

                # Create results DataFrame for chatbot analysis
                lr_results_df = pd.DataFrame({
                    "date": dates,
                    "actual": y_true,
                    "predicted": y_pred
                })
                chatbot_ui(lr_results_df, f"lr_{target_ticker}", chatbot)

                # 5. Show Inference for Next Day
                pred_price, pred_date = linear_regression_service.load_and_predict(
                    final_df, target_ticker, model_path, scaler_path, is_test=False
                )
                st.success(f"Predicted Next-Day Close Price for {target_ticker} on {pd.to_datetime(pred_date).date() + timedelta(days=1)}: **${pred_price:.2f}**")

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
                # Define model and scaler paths
                rf_model_path = os.path.join(Config.MODELS_DIR, f"rf_model_{target_ticker}.joblib")
                rf_scaler_path = os.path.join(Config.MODELS_DIR, f"rf_scaler_{target_ticker}.joblib")
                
                # Replicate the logic from the service to get the feature names
                target_col = f"{target_ticker}_close_stock"
                features_to_drop = ['Date', 'target_diff'] + [col for col in final_df.columns if col.endswith('_close_stock')]
                feature_names = [col for col in final_df.columns if col not in features_to_drop]

                # 1. Process Data (unpacking 7 values now)
                X_train, _, y_train, _, scaler, _, _ = random_forest_service.process_rf_features(final_df, target_ticker, is_training=True)
                st.write("✅ Data processed for Random Forest.")

                # 2. Train and Save Model
                random_forest_service.train_and_save_rf_model(X_train, y_train, scaler, rf_model_path, rf_scaler_path)
                st.write(f"✅ Random Forest model and scaler for {target_ticker} trained and saved.")

                # 3. Evaluate on Test Set and Plot
                y_pred_prices, y_actual_prices, dates = random_forest_service.load_rf_model_and_predict(final_df, target_ticker, rf_model_path, rf_scaler_path, is_test=True)
                fig_rf = random_forest_service.plot_rf_prediction(y_actual_prices, y_pred_prices, dates, title=f"RF: Actual vs. Predicted for {target_ticker}")
                st.pyplot(fig_rf)

                # Create results DataFrame for chatbot analysis
                rf_results_df = pd.DataFrame({
                    "date": dates,
                    "actual": y_actual_prices,
                    "predicted": y_pred_prices
                })
                chatbot_ui(rf_results_df, f"rf_{target_ticker}", chatbot)

                # 5. Show Inference for Next Day
                pred_price_rf = random_forest_service.load_rf_model_and_predict(final_df, target_ticker, rf_model_path, rf_scaler_path, is_test=False)
                last_known_date = pd.to_datetime(final_df['Date']).iloc[-1]
                st.success(f"RF Predicted Next-Day Close for {target_ticker} on {last_known_date.date() + timedelta(days=1)}: **${pred_price_rf:.2f}**")

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

                # 1. Process Data
                X_train, y_train, _, _, scaler_X, scaler_y, _, _ = lstm_service.process_lstm_data(
                    final_df, target_col_lstm, sequence_length=seq_length, is_training=True
                )
                st.write("✅ Data processed for LSTM.")

                # 2. Train and Save Model
                lstm_service.train_and_save_lstm(
                    X_train, y_train, scaler_X, scaler_y, lstm_model_path, lstm_scaler_path, sequence_length=seq_length
                )
                st.write(f"✅ LSTM model and scalers for {target_ticker} trained and saved.")

                # 3. Evaluate on Test Set and Plot
                y_pred, y_true, dates = lstm_service.load_and_predict_lstm(
                    final_df, target_col_lstm, lstm_model_path, lstm_scaler_path, sequence_length=seq_length, is_test=True
                )
                fig_lstm = lstm_service.plot_lstm_results(y_true, y_pred, dates, title=f"LSTM: Actual vs. Predicted for {target_ticker}")
                st.pyplot(fig_lstm)

                # Create results DataFrame for chatbot analysis
                lstm_results_df = pd.DataFrame({
                    "date": dates,
                    "actual": y_true,
                    "predicted": y_pred
                })
                chatbot_ui(lstm_results_df, f"lstm_{target_ticker}", chatbot)

                # 4. Show Inference for Next Day
                pred_price_lstm, pred_date_lstm = lstm_service.load_and_predict_lstm(
                    final_df, target_col_lstm, lstm_model_path, lstm_scaler_path, sequence_length=seq_length, is_test=False
                )
                st.success(f"LSTM Predicted Next-Day Close for {target_ticker} on {pd.to_datetime(pred_date_lstm).date() + timedelta(days=1)}: **${pred_price_lstm:.2f}**")

    st.button("↩️ Restart Pipeline", on_click=set_step, args=[0])
