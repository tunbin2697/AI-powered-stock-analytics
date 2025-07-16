# T-GNN++ for Multi-Asset Financial Forecasting

## What is T-GNN++?

**T-GNN++** (Temporal Graph Neural Network++) is an advanced deep learning architecture for financial time series forecasting. It leverages both **spatial relationships** (between multiple assets/stocks) and **temporal dependencies** (across time) to predict the next-day price movement of a target stock. T-GNN++ is especially powerful for multi-modal data, integrating stock prices, news sentiment, and macroeconomic indicators.

---

## T-GNN++ Architecture Overview

The T-GNN++ model consists of three main modules:

1. **Spatial Module (GAT):**

   - Uses a Graph Attention Network (GAT) to model inter-stock (cross-asset) relationships at each time step.
   - Each node (stock) attends to its correlated peers, capturing market structure.

2. **Short-Term Temporal Module (GRU):**

   - For each stock, a Gated Recurrent Unit (GRU) models its own short-term temporal evolution.
   - This allows the model to remember recent patterns for each asset.

3. **Long-Term Temporal Module (Transformer):**

   - A Transformer Encoder captures long-range dependencies across the time window for each stock.
   - This module enables the model to focus on the most relevant days in the past for prediction.

4. **Prediction Head:**
   - Only the representation of the **target stock** is used to predict its next-day price difference (or return).

**Pipeline:**  
`[Features for all stocks, news, macro] → GAT (spatial) → GRU (temporal) → Transformer (long-term) → Predict target stock`

---

## Code Structure and Key Functions

This section explains the main code components, their purpose, and their input/output.

### 1. Model Definition: `TGNNPP` class

- **Purpose:** Implements the T-GNN++ architecture.
- **Inputs:**
  - `node_feats`: Tensor of shape `[batch_size, window_size, num_nodes, feature_dim]`
  - `edge_indices`: List of edge index tensors for each sample in the batch
  - `return_attention`: If `True`, returns attention weights for XAI
- **Outputs:**
  - If `return_attention=False`: Predicted next-day price difference for the target stock
  - If `return_attention=True`: Tuple of (prediction, GAT attention weights, Transformer attention weights)
- **Special Features:**
  - Handles dynamic graph construction for each time window.
  - Returns attention weights for explainability (DAVOTS, ICFTS).
  - Only the target stock's representation is used for prediction.

### 2. Data Preparation Functions

#### `generate_finbert_embeddings(news_df, batch_size=16)`

- **Purpose:** Converts news titles into FinBERT embeddings for each stock and day.
- **Input:** `news_df` (DataFrame with columns like `AAPL_title`, `MSFT_title`, etc.)
- **Output:** Numpy array of shape `[num_days, num_stocks, 768]` (FinBERT embedding size)
- **Reason:** News sentiment is a key driver of stock prices; embeddings allow the model to use this information numerically.

#### `scale_features(stock_df, macro_df)`

- **Purpose:** Scales stock and macroeconomic features using MinMaxScaler.
- **Input:** `stock_df`, `macro_df` (DataFrames)
- **Output:** Scaled DataFrames and fitted scalers
- **Reason:** Scaling ensures all features are on a comparable scale, improving model convergence.

### 3. Dataset Construction

#### `create_tgnn_dataset(unscaled_stock_df, scaled_stock_df, scaled_macro_df, news_embeddings, target_stock, window_size=30, corr_threshold=0.5)`

- **Purpose:** Builds the dataset for T-GNN++ by combining all features and constructing dynamic graphs.
- **Inputs:**
  - Unscaled and scaled stock DataFrames
  - Scaled macro DataFrame
  - News embeddings
  - Target stock symbol
  - Window size (number of days for each sample)
  - Correlation threshold for graph construction
- **Outputs:**
  - `dataset`: List of tuples `(node_features, edge_index, target_diff, meta_info)`
  - `stock_codes`: List of stock symbols
  - `feature_dim`: Number of features per node
- **Reason:** Each sample contains all information needed for the model: node features, graph structure, and the prediction target.

### 4. Training and Evaluation

#### `train_tgnn_model(dataset, feature_dim, num_stocks, target_stock_index, model_save_path, epochs=20, lr=1e-4, batch_size=16)`

- **Purpose:** Trains the T-GNN++ model and saves the best version.
- **Inputs:**
  - Dataset and model hyperparameters
- **Output:**
  - Saves the trained model to disk
- **Reason:** Handles batching, loss calculation, and validation for robust training.

#### `test_and_plot_tgnn(test_dataset, feature_dim, num_stocks, target_stock_index, model_path)`

- **Purpose:** Loads a trained model, evaluates it on the test set, and plots actual vs. predicted prices.
- **Inputs:**
  - Test dataset and model parameters
- **Output:**
  - Plots and prints evaluation metrics
- **Reason:** Visualizes model performance for interpretation.

### 5. XAI Visualization

#### `visualize_icfts_gat_attention(model, data_sample, stock_codes, time_step_to_viz=-1)`

- **Purpose:** Visualizes GAT attention scores (ICFTS) to show which stocks influenced the target at a specific time step.
- **Inputs:**
  - Trained model, a data sample, stock codes, and time step
- **Output:**
  - Heatmap plot of inter-stock attention
- **Reason:** Provides explainability for the model's spatial reasoning.

#### `visualize_davots_transformer_attention(model, data_sample, window_size)`

- **Purpose:** Visualizes Transformer attention (DAVOTS) to show which days in the window were most important for the prediction.
- **Inputs:**
  - Trained model, a data sample, window size
- **Output:**
  - Bar plot of temporal attention
- **Reason:** Provides explainability for the model's temporal reasoning.

---

## Special Features and Dedicated Code Sections

- **Dynamic Graph Construction:**  
  The graph structure (edges) is rebuilt for each sample based on rolling correlations, reflecting changing market relationships.

- **Multi-Modal Feature Integration:**  
  The code combines price, volume, macro, and news sentiment into a single feature tensor for each node.

- **Explainable AI (XAI):**  
  The model is designed to return attention weights for both spatial (ICFTS) and temporal (DAVOTS) explainability, with dedicated visualization functions.

- **Targeted Prediction:**  
  Only the target stock's representation is used for prediction, even though the model processes all stocks.

---

## How to Build the T-GNN++ Dataset

### 1. Data Sources

- **Stock Data:** Historical prices and volumes for multiple stocks.
- **News Data:** Daily news titles for each stock, processed into sentiment embeddings using FinBERT.
- **Macro Data:** Daily macroeconomic indicators (e.g., interest rates, indices).

### 2. Data Preprocessing & Feature Engineering

- **Merge:** All data sources are merged on the date.
- **Fill NaNs:**
  - Stock columns: Rolling mean fill, then forward/backward fill.
  - News sentiment: Fill missing with 0 or empty string.
  - Macro: Forward fill.
- **Scaling:** Stock and macro features are scaled using MinMaxScaler (except for the target variable).
- **News Embedding:** News titles are converted to 768-dimensional FinBERT embeddings for each stock and day.

### 3. Dataset Construction

- For each day, a **window** of past days is used to create a sample.
- **Node features:** For each stock, concatenate its scaled features, macro features, and news embeddings for each day in the window.
- **Graph structure:** Dynamically constructed using rolling window correlations between stocks.
- **Target:** The next day's percentage change in close price for the target stock.

---

## Why Feature the Dataset This Way?

- **Multi-Asset Learning:** By including all stocks, the model can learn how assets influence each other (market structure).
- **Multi-Modal Inputs:** Combining price, news, and macro data allows the model to capture more complex drivers of price movement.
- **Temporal Windows:** Using a window of past days enables the model to learn both short-term and long-term dependencies.
- **Dynamic Graphs:** Building the graph from rolling correlations allows the model to adapt to changing market relationships.
- **Target as Percentage Change:** Predicting returns (instead of raw price) stabilizes the learning process and makes the model robust to price scale changes.

---

## How to Run

1. **Prepare your data:**

   - `featured_stock_data`, `news_raw_data`, and `ffilled_macro_data` should be loaded as pandas DataFrames.

2. **Run the pipeline:**

   - The main block in `tgnn_service.py` will:
     - Merge and fill data
     - Scale features
     - Generate FinBERT embeddings
     - Build the dataset
     - Train and test the T-GNN++ model
     - Visualize XAI explanations (ICFTS, DAVOTS)

3. **Interpret results:**
   - The model will output prediction plots and XAI visualizations to help you understand both the accuracy and the reasoning behind each prediction.

---

## References

- [T-GNN++ Paper](https://arxiv.org/abs/2302.06653)
- [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903)
- [Transformer Encoder](https://arxiv.org/abs/1706.03762)
- [FinBERT](https://github.com/ProsusAI/finBERT)
