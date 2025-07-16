import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import Config
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Sentiment Utilities
def get_finbert():
    if not hasattr(get_finbert, "tokenizer"):
        get_finbert.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        get_finbert.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
    return get_finbert.tokenizer, get_finbert.model

tokenizer, sentiment_model = get_finbert()

def analyze_sentiment(text):
    if pd.isna(text):
        return {'positive': 0, 'negative': 0, 'neutral': 0}
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**tokens)
    scores = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    return {'positive': scores[0], 'negative': scores[1], 'neutral': scores[2]}

def sentiment_3_class_news_data(news_df):
    df = news_df.copy()
    for col in df.columns:
        if col.endswith('_title'):
            ticker = col.replace('_title', '')
            s = df[col].apply(analyze_sentiment)
            df[f"{ticker}_sentiment_positive"] = s.apply(lambda x: x['positive'])
            df[f"{ticker}_sentiment_negative"] = s.apply(lambda x: x['negative'])
            df[f"{ticker}_sentiment_neutral"]  = s.apply(lambda x: x['neutral'])
    return df.drop(columns=[c for c in df.columns if c.endswith('_title')])

# 2. Merge & Fill
def merge_and_fill_nan(stock_df, news_df, macro_df, date_col='Date', rolling_window=3):
    dfs = [stock_df.copy(), news_df.copy(), macro_df.copy()]
    for d in dfs:
        d[date_col] = pd.to_datetime(d[date_col])
    merged = dfs[0].merge(dfs[1], on=date_col, how='outer')
    merged = merged.merge(dfs[2], on=date_col, how='outer')
    merged.sort_values(date_col, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    for col in merged.columns:
        if col == date_col:
            continue
        if col.endswith('_stock'):
            merged[col] = merged[col].fillna(
                merged[col].rolling(rolling_window, min_periods=1, center=True).mean()
            ).ffill().bfill()
        elif '_sentiment_' in col:
            merged[col] = merged[col].fillna(0)
        elif col.endswith('_macro'):
            merged[col] = merged[col].ffill()
    # Final clean: replace inf/nan
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.fillna(0)
    return merged

# 3. Model Definition
# -------------------------------
# 0. T-GNN++ MODEL ARCHITECTURE
# -------------------------------
class TGNNPP(nn.Module):
    """T-GNN++ Transformer Architecture with integrated embedding"""
    def __init__(self, num_nodes, feature_dim, window_size=30, embed_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size

        # Feature embedding layer
        self.feature_embed = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim))

        # T-GNN Module (GAT + Temporal Attention)
        self.gat = GATConv(embed_dim, 64, heads=2, concat=True)
        self.temp_attention = nn.MultiheadAttention(128, num_heads=4)

        # Transformer Module
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=4
        )

        # Fusion and Prediction
        self.fusion = nn.Linear(256, 128)
        self.pred_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, node_feats, edge_index):
        """Process windowed stock data"""
        # node_feats: [batch_size, window_size, num_nodes, feature_dim]
        batch_size, window_size, num_nodes, feature_dim = node_feats.shape

        # Reshape for embedding: [batch*window*nodes, features]
        embedded = self.feature_embed(
            node_feats.view(batch_size * window_size * num_nodes, -1))
        embedded = embedded.view(batch_size, window_size, num_nodes, -1)

        # Process each time step with GAT
        gat_outputs = []
        for b in range(batch_size):
            batch_outputs = []
            for t in range(window_size):
                # Extract 2D tensor for GAT: [num_nodes, embed_dim]
                x = embedded[b, t, :, :]

                # Apply GAT (expects 2D input)
                gat_out = self.gat(x, edge_index)  # [num_nodes, 128] (64*2 heads)
                batch_outputs.append(gat_out.unsqueeze(0))

            # Stack temporal outputs: [1, window_size, num_nodes, 128]
            batch_temporal = torch.cat(batch_outputs, dim=0).unsqueeze(0)
            gat_outputs.append(batch_temporal)

        # Combine all batches: [batch_size, window_size, num_nodes, 128]
        temporal = torch.cat(gat_outputs, dim=0)

        # Reshape for temporal attention: [window_size, batch_size, features]
        temporal_reshaped = temporal.mean(dim=2)  # Average over nodes: [batch, window, 128]
        temporal_reshaped = temporal_reshaped.permute(1, 0, 2)  # [window, batch, 128]

        # Temporal attention
        attn_output, _ = self.temp_attention(
            temporal_reshaped,
            temporal_reshaped,
            temporal_reshaped
        )

        # Transformer processing
        transformer_output = self.transformer(temporal_reshaped)

        # Take last timestep and average
        attn_output = attn_output[-1]  # [batch, 128]
        transformer_output = transformer_output[-1]  # [batch, 128]

        # Fusion
        fused = torch.cat([attn_output, transformer_output], dim=-1)
        fused = F.relu(self.fusion(fused))

        # Predict
        return self.pred_head(fused).squeeze()

# 4. Dataset Creation
def create_tgnn_dataset(stock_df, macro_df, news_embeddings, target_stock):
    """Combine all data sources for T-GNN++"""
    news_embeddings = np.load(news_embeddings) if isinstance(news_embeddings, str) else news_embeddings

    # Get stock codes
    stock_codes = list({col.split('_')[0] for col in stock_df.columns if '_stock' in col})
    num_stocks = len(stock_codes)
    num_days = len(stock_df)

    # Create feature tensor
    feature_tensor = np.zeros((num_days, num_stocks, 0), dtype=np.float32)

    # 1. Add stock features
    stock_features = []
    for code in stock_codes:
        code_cols = [col for col in stock_df.columns if col.startswith(f"{code}_") and '_stock' in col]
        stock_features.append(stock_df[code_cols].values)
    stock_features = np.stack(stock_features, axis=1)
    feature_tensor = np.concatenate([feature_tensor, stock_features], axis=-1)

    # 2. Add macro features (broadcast to all stocks)
    macro_features = macro_df.drop(columns=['Date']).values
    macro_features = np.repeat(macro_features[:, np.newaxis, :], num_stocks, axis=1)
    feature_tensor = np.concatenate([feature_tensor, macro_features], axis=-1)

    # 3. Add news embeddings
    feature_tensor = np.concatenate([feature_tensor, news_embeddings], axis=-1)

    # Create dataset with dynamic graphs
    dataset = []
    window_size = 30
    target_idx = stock_codes.index(target_stock)

    for day_idx in range(window_size, num_days):

        # Dynamic graph construction
        returns = []
        for stock_idx in range(num_stocks):
            close_col = f"{stock_codes[stock_idx]}_close_stock"
            prices = stock_df[close_col].iloc[day_idx-window_size:day_idx].values
            log_returns = np.diff(np.log(prices))
            returns.append(log_returns)

        corr_matrix = np.corrcoef(returns)
        adj_matrix = (corr_matrix > 0.5).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        edge_index = torch.tensor(np.where(adj_matrix), dtype=torch.long)

        # Node features (window)
        node_features = feature_tensor[day_idx-window_size:day_idx]

        # Target (next day close price)
        target = stock_df[f"{target_stock}_close_stock"].iloc[day_idx]

        dataset.append((node_features, edge_index, target))

    return dataset, stock_codes, feature_tensor.shape[-1]

# 5. Train/Test with XAI
def train_tgnn(dataset, feature_dim, num_stocks, epochs=50, lr=1e-4):
    """Train T-GNN++ model"""
    # Split dataset
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGNNPP(
        num_nodes=num_stocks,
        feature_dim=feature_dim,
        window_size=30
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for node_feats, edge_index, target in train_data:
            # Move data to device
            node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
            edge_index = edge_index.to(device)
            target = torch.tensor([target], dtype=torch.float32).to(device)

            # Forward pass
            optimizer.zero_grad()
            pred = model(node_feats, edge_index)
            loss = criterion(pred, target)

            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_data):.6f}")

    # Test model
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for node_feats, edge_index, target in test_data:
            node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
            edge_index = edge_index.to(device)
            pred = model(node_feats, edge_index)

            test_preds.append(pred.cpu().item())
            test_targets.append(target)

    # Calculate metrics
    mse = np.mean((np.array(test_preds) - np.array(test_targets))**2)
    print(f"\nTest MSE: {mse:.6f}")

    return model

# 6. XAI Visualization Functions
# def plot_gat_attention(model, sample, codes):
#     x, ei, _ = sample
#     y, (e_idx, gat_w), _ = model(torch.tensor(x[None],dtype=torch.float32).to(next(model.parameters()).device),
#                                  ei.to(next(model.parameters()).device), return_attn=True)
#     att = gat_w.mean(dim=0).cpu().numpy()
#     M = len(codes); mat = np.zeros((M, M))
#     idx = e_idx.cpu().numpy()
#     mat[idx[0], idx[1]] = att
#     plt.figure(figsize=(6,5)); plt.matshow(mat, fignum=0); plt.colorbar()
#     plt.xticks(range(M), codes, rotation=90); plt.yticks(range(M), codes)
#     plt.title('GAT Attention'); plt.show()

# def plot_temp_attention(model, sample, window_size):
#     x, ei, _ = sample
#     y, (_, _), temp_w = model(torch.tensor(x[None],dtype=torch.float32).to(next(model.parameters()).device),
#                                ei.to(next(model.parameters()).device), return_attn=True)
#     att = temp_w.mean(dim=1).cpu().numpy()[0]
#     plt.figure(figsize=(8,3)); plt.bar(range(window_size), att)
#     plt.xlabel('Timestep'); plt.ylabel('Attention'); plt.title('Temporal Attention'); plt.show()

def sentiment_df_to_news_embeddings(sentiment_df, embedding_dim=3):
    """
    Adapter function to convert sentiment DataFrame to 3D numpy array
    with shape (days, stocks, embedding_dim), matching the output of
    `generate_finbert_embeddings`.

    Args:
        sentiment_df (pd.DataFrame): The sentiment-enhanced news DataFrame,
                                     output from `sentiment_3_class_news_data`.
        embedding_dim (int): Use 3 for (pos, neg, neu) or 5 if including polarity/confidence.

    Returns:
        np.ndarray: (days, stocks, embedding_dim)
    """
    # Extract stock codes from columns
    sentiment_cols = [col for col in sentiment_df.columns if '_sentiment_' in col]
    stock_codes = sorted(set(col.split('_sentiment_')[0] for col in sentiment_cols))
    num_days = len(sentiment_df)

    # Initialize the output embeddings
    news_embeddings = np.zeros((num_days, len(stock_codes), embedding_dim), dtype=np.float32)

    for stock_idx, stock in enumerate(stock_codes):
        pos_col = f"{stock}_sentiment_positive"
        neg_col = f"{stock}_sentiment_negative"
        neu_col = f"{stock}_sentiment_neutral"

        if embedding_dim == 3:
            vectors = sentiment_df[[pos_col, neg_col, neu_col]].values
        elif embedding_dim == 5:
            polarity = sentiment_df[pos_col] - sentiment_df[neg_col]
            confidence = sentiment_df[[pos_col, neg_col, neu_col]].max(axis=1)
            vectors = np.column_stack([
                sentiment_df[[pos_col, neg_col, neu_col]].values,
                polarity.values,
                confidence.values
            ])
        else:
            raise ValueError("Only embedding_dim=3 or 5 is supported.")

        news_embeddings[:, stock_idx, :] = vectors

    return news_embeddings

stock_scaler = StandardScaler()
macro_scaler = StandardScaler()

def scale_tgnn_data(stock_df, macro_df):
  """
  Standard scales the stock and macro dataframes and saves the scalers.

  Args:
    stock_df (pd.DataFrame): DataFrame with stock data.
    macro_df (pd.DataFrame): DataFrame with macro data.

  Returns:
    tuple: A tuple containing the scaled stock DataFrame and the scaled macro DataFrame.
  """
  # Fit and transform stock data, excluding 'Date'
  stock_cols_to_scale = [col for col in stock_df.columns if col != 'Date']
  scaled_stock_data = stock_scaler.fit_transform(stock_df[stock_cols_to_scale])
  scaled_stock_df = pd.DataFrame(scaled_stock_data, columns=stock_cols_to_scale, index=stock_df.index)
  scaled_stock_df['Date'] = stock_df['Date'] # Add Date back
  scaled_stock_df = scaled_stock_df[['Date'] + stock_cols_to_scale] # Reorder Date to front

  # Fit and transform macro data, excluding 'Date'
  macro_cols_to_scale = [col for col in macro_df.columns if col != 'Date']
  scaled_macro_data = macro_scaler.fit_transform(macro_df[macro_cols_to_scale])
  scaled_macro_df = pd.DataFrame(scaled_macro_data, columns=macro_cols_to_scale, index=macro_df.index)
  scaled_macro_df['Date'] = macro_df['Date'] # Add Date back
  scaled_macro_df = scaled_macro_df[['Date'] + macro_cols_to_scale] # Reorder Date to front

  return scaled_stock_df, scaled_macro_df

if __name__=="__main__":

    daily_data = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv"))
    
    featured_stock_data = daily_data[[col for col in daily_data.columns if col.endswith("_stock") or col == "Date"]]
    news_sent = daily_data[[col for col in daily_data.columns if "sentiment" in col or col == "Date"]]
    ffilled_macro_data = daily_data[[col for col in daily_data.columns if col.endswith("_macro") or col == "Date"]]
    
    news_embeddings = sentiment_df_to_news_embeddings(news_sent, embedding_dim=3)

    scaled_stock_df, scaled_macro_df = scale_tgnn_data(featured_stock_data, ffilled_macro_data)
    
    dataset, stock_codes, feature_dim = create_tgnn_dataset(
        stock_df= scaled_stock_df,
        macro_df= scaled_macro_df,
        news_embeddings=news_embeddings,
        target_stock="AAPL"
    )
    
    model = train_tgnn(
        dataset=dataset,
        feature_dim=feature_dim,
        num_stocks=len(stock_codes),
        epochs=75
    )
    
    torch.save(model.state_dict(), "tgnnpp_finbert_model.pth")
    
    print("\n--- Testing Section ---")

    # Reload the model if necessary
    # model = TGNNPP(num_nodes=len(["AAPL","TSLA"]), feature_dim=feature_dim, window_size=30)
    model.load_state_dict(torch.load("tgnnpp_finbert_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode

    split_idx = int(0.8 * len(dataset))
    test_data = dataset[split_idx:]

    test_preds, test_targets = [], []
    with torch.no_grad():
        for node_feats, edge_index, target in test_data:
            node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
            edge_index = edge_index.to(device)
            pred = model(node_feats, edge_index)

            test_preds.append(pred.cpu().item())
            test_targets.append(target)

    # Calculate test metrics
    mse = np.mean((np.array(test_preds) - np.array(test_targets))**2)
    print(f"Final Test MSE: {mse:.6f}")

    # --- Visualization ---
    print("\n--- Visualization ---")

    # Plot actual vs. predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(test_targets, label='Actual Price', marker='o', linestyle='-')
    plt.plot(test_preds, label='Predicted Price', marker='x', linestyle='--')
    plt.title(f'{dataset[0][0][0, 0].shape[0]}_stock Actual vs. Predicted Prices')
    plt.xlabel('Time Steps (in Test Set)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()