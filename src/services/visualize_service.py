import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class VisualizeService:

    @staticmethod
    def create_rsi_figs(df):
        """
        Extract RSI14 and RSI21 data from a DataFrame and return Plotly line chart figures.

        Assumes columns are named like 'AAPL_rsi14', 'TSLA_rsi21', etc.

        Args:
            df (pd.DataFrame): DataFrame with 'Date' and RSI columns.

        Returns:
            list[tuple[str, plotly.graph_objs.Figure]]: (title, figure) pairs for Streamlit.
        """
        df = df.copy()

        # Only accept exact rsi14 or rsi21
        valid_rsi_types = ["rsi14", "rsi21"]
        rsi_cols = [
            col for col in df.columns
            if any(col.lower().endswith(f"_{rsi_type}_stock") for rsi_type in valid_rsi_types)
        ]

        result = {}
        for col in rsi_cols:
            try:
                stock, rsi_type = col.split('_')
            except ValueError:
                continue  # Ignore malformed column names

            if rsi_type not in valid_rsi_types:
                continue

            if stock not in result:
                result[stock] = {}

            series = [
                {"date": date, "value": value}
                for date, value in zip(df["Date"], df[col])
            ]
            result[stock][rsi_type] = series

        figures = []
        for stock, rsi_dict in result.items():
            for rsi_type, series in rsi_dict.items():
                if not series:
                    continue
                df_plot = pd.DataFrame(series)
                df_plot["date"] = pd.to_datetime(df_plot["date"])
                fig = px.line(
                    df_plot,
                    x="date",
                    y="value",
                    title=f"{stock} {rsi_type.upper()}",
                    labels={"date": "Date", "value": "RSI Value"},
                )
                fig.update_layout(
                    yaxis=dict(range=[0, 100]),
                    title_x=0.5,
                    template="plotly_white",
                    height=400
                )
                fig.add_shape(
                    type="line", x0=df_plot['date'].min(), x1=df_plot['date'].max(),
                    y0=70, y1=70, line=dict(color="red", dash="dash")
                )
                fig.add_shape(
                    type="line", x0=df_plot['date'].min(), x1=df_plot['date'].max(),
                    y0=30, y1=30, line=dict(color="blue", dash="dash")
                )
                figures.append((f"{stock} {rsi_type.upper()}", fig, df_plot))
        return figures

    @staticmethod
    def create_ma_figs(df):
        """
        Extract MA50 and MA200 from the DataFrame and return Plotly line chart figures.

        Assumes columns are named like 'AAPL_ma50', 'TSLA_ma200', etc.

        Args:
            df (pd.DataFrame): DataFrame with 'Date' and stock MA columns.

        Returns:
            list[tuple[str, plotly.graph_objs.Figure]]: (title, figure) pairs for Streamlit.
        """
        df = df.copy()

        # Target only specific MA types
        ma_targets = ["ma50", "ma200"]
        ma_cols = [col for col in df.columns if any(col.endswith(f"_{target}_stock") for target in ma_targets)]

        result = {}
        for col in ma_cols:
            try:
                stock, ma_type = col.split('_')
            except ValueError:
                continue  # Skip if format is wrong

            if stock not in result:
                result[stock] = {}

            series = [
                {"date": date, "value": value}
                for date, value in zip(df["Date"], df[col])
            ]
            result[stock][ma_type] = series

        figures = []
        for stock, ma_dict in result.items():
            for ma_type, series in ma_dict.items():
                if not series:
                    continue
                df_plot = pd.DataFrame(series)
                fig = px.line(
                    df_plot,
                    x="date",
                    y="value",
                    title=f"{stock} {ma_type.upper()}",
                    labels={"date": "Date", "value": "MA Value"},
                )
                fig.update_layout(
                    title_x=0.5,
                    template="plotly_white",
                    height=400
                )
                figures.append((f"{stock} {ma_type.upper()}", fig, df_plot))
        return figures
    
    @staticmethod
    def create_ohlcv_fig(df, ticker):
        """
        Create a Plotly candlestick + volume chart for a given ticker.

        Args:
            df (pd.DataFrame): DataFrame with columns like '{ticker}_open_stock', ..., and 'Date'.
            ticker (str): Stock symbol to extract (e.g., 'AAPL').

        Returns:
            plotly.graph_objs.Figure | None: The OHLCV chart figure, or None if data is missing.
        """
        open_col = f'{ticker}_open_stock'
        high_col = f'{ticker}_high_stock'
        low_col = f'{ticker}_low_stock'
        close_col = f'{ticker}_close_stock'
        volume_col = f'{ticker}_volume_stock'
        date_col = 'Date'

        required_cols = [date_col, open_col, high_col, low_col, close_col, volume_col]
        if not all(col in df.columns for col in required_cols):
            return None

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, open_col, high_col, low_col, close_col, volume_col]].dropna()
        df.sort_values(by=date_col, inplace=True)

        if df.empty:
            return None, None

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df[date_col],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col],
            name="Candlestick"
        ))

        fig.add_trace(go.Bar(
            x=df[date_col],
            y=df[volume_col],
            name='Volume',
            yaxis='y2',
            marker_color='rgba(0, 0, 255, 0.3)'
        ))

        fig.update_layout(
            title=f"{ticker} OHLCV Chart",
            xaxis=dict(title="Date", rangeslider=dict(visible=False)),
            yaxis=dict(title="Price", domain=[0.2, 1.0]),
            yaxis2=dict(title="Volume", overlaying='y', side='right', domain=[0.0, 0.2]),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            template="plotly_white",
            height=600
        )

        return fig, df
    
    @staticmethod
    def create_missing_value_bar_chart(df, title="Missing Values per Column"):
        """
        Create a bar chart showing count and percentage of missing values per column.
        """
        missing_count = df.isnull().sum()
        missing_percent = 100 * missing_count / len(df)
        missing_info = pd.DataFrame({'count': missing_count, 'percent': missing_percent})
        missing_info = missing_info[missing_info['count'] > 0].sort_values(by='count', ascending=False)

        if missing_info.empty:
            return None, None

        fig = px.bar(
            missing_info,
            y='count',
            x=missing_info.index,
            text='percent',
            title=title,
            labels={'index': 'Column', 'count': 'Missing Count', 'percent': 'Missing (%)'}
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        return fig, missing_info

    @staticmethod
    def create_daily_return_histogram(df, ticker):
        """
        Create histogram for the distribution of daily returns for a given stock.
        """
        return_col = f'{ticker}_daily_return_stock'
        if return_col not in df.columns:
            return None, None

        returns_data = df[return_col].dropna()
        if returns_data.empty:
            return None, None

        fig = px.histogram(
            returns_data,
            nbins=50,
            title=f'Distribution of Daily Returns for {ticker}',
            labels={'value': 'Daily Return'},
            template='plotly_white'
        )
        fig.update_layout(bargap=0.1)
        return fig, returns_data.to_frame(name='daily_return')

    @staticmethod
    def create_sentiment_line_chart(df, ticker):
        """
        Create line chart showing sentiment trends over time (positive, negative, neutral).
        """
        date_col = 'Date'
        sentiment_cols = [
            f'{ticker}_sentiment_positive',
            f'{ticker}_sentiment_negative',
            f'{ticker}_sentiment_neutral'
        ]

        if not all(col in df.columns for col in sentiment_cols):
            return None, None

        sentiment_df = df[[date_col] + sentiment_cols].copy()
        sentiment_df.dropna(subset=sentiment_cols, how='all', inplace=True)
        sentiment_df.sort_values(by=date_col, inplace=True)

        if sentiment_df.empty:
            return None, None

        sentiment_melted = sentiment_df.melt(
            id_vars=[date_col],
            value_vars=sentiment_cols,
            var_name='Sentiment',
            value_name='Score'
        )
        sentiment_melted['Sentiment'] = sentiment_melted['Sentiment'].str.replace(f'{ticker}_sentiment_', '')

        fig = px.line(
            sentiment_melted,
            x=date_col,
            y='Score',
            color='Sentiment',
            title=f'Sentiment Scores Over Time for {ticker}',
            labels={date_col: 'Date', 'Score': 'Sentiment Score'},
            template='plotly_white'
        )
        fig.update_layout(yaxis=dict(range=[0, 1]))
        return fig, sentiment_melted

    @staticmethod
    def create_macro_timeseries_line_chart(df, series_id):
        """
        Create a line chart for a macroeconomic time series (e.g., FEDFUNDS).
        """
        date_col = 'Date'
        macro_col = f'{series_id}_macro'
        if macro_col not in df.columns:
            return None, None

        macro_df = df[[date_col, macro_col]].copy()
        macro_df.dropna(subset=[macro_col], inplace=True)
        macro_df.sort_values(by=date_col, inplace=True)

        if macro_df.empty:
            return None, None

        fig = px.line(
            macro_df,
            x=date_col,
            y=macro_col,
            title=f'{series_id} Macroeconomic Data Over Time',
            labels={date_col: 'Date', macro_col: 'Value'},
            template='plotly_white'
        )
        return fig, macro_df

    @staticmethod
    def create_correlation_heatmap(df, title="Correlation Heatmap"):
        """
        Create a heatmap showing correlations between numerical columns.
        """
        numerical_df = df.select_dtypes(include=np.number)
        if numerical_df.empty:
            return None, None

        correlation_matrix = numerical_df.corr()

        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            title=title,
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        return fig, correlation_matrix
    
    @staticmethod
    def create_news_wordcloud_figure(df, ticker):
        """
        Generate a word cloud figure from news titles for a specific ticker.

        Args:
            df (pd.DataFrame): The merged DataFrame containing news titles.
            ticker (str): Stock ticker symbol (e.g., 'AAPL').

        Returns:
            matplotlib.figure.Figure or None: WordCloud figure or None if data is invalid.
        """
        title_col = f'{ticker}_title'

        if title_col not in df.columns:
            return None, None

        all_titles = " ".join(df[title_col].dropna().tolist())

        if not all_titles.strip():
            return None, None

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(all_titles)

        # Create and return the figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Word Cloud for {ticker} News Titles")
        # Return the figure and the DataFrame of titles used
        return fig, df[[title_col]].dropna()