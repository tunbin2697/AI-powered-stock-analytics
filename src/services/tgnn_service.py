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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- TGNNPP Model Definition (unchanged) ---
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

# --- 1. Prepare Data ---
def prepare_tgnn_data(daily_data, target_stock="AAPL", embedding_dim=3):
    """
    Prepare scaled stock, macro, and news embeddings for TGNN.
    Returns: dataset, stock_codes, feature_dim, test_split_idx
    """
    featured_stock_data = daily_data[[col for col in daily_data.columns if col.endswith("_stock") or col == "Date"]]
    news_sent = daily_data[[col for col in daily_data.columns if "sentiment" in col or col == "Date"]]
    ffilled_macro_data = daily_data[[col for col in daily_data.columns if col.endswith("_macro") or col == "Date"]]

    # Scale stock features
    stock_cols = [col for col in featured_stock_data.columns if col.endswith("_stock") and col != "Date"]
    scaler_stock = StandardScaler()
    featured_stock_data.loc[:, stock_cols] = scaler_stock.fit_transform(featured_stock_data[stock_cols])

    # Scale macro features
    macro_cols = [col for col in ffilled_macro_data.columns if col.endswith("_macro") and col != "Date"]
    scaler_macro = StandardScaler()
    ffilled_macro_data.loc[:, macro_cols] = scaler_macro.fit_transform(ffilled_macro_data[macro_cols])

    # News embeddings
    news_embeddings = sentiment_df_to_news_embeddings(news_sent, embedding_dim=embedding_dim)

    # Create TGNN dataset
    dataset, stock_codes, feature_dim = create_tgnn_dataset(
        stock_df=featured_stock_data,
        macro_df=ffilled_macro_data,
        news_embeddings=news_embeddings,
        target_stock=target_stock
    )
    split_idx = int(0.8 * len(dataset))
    return dataset, stock_codes, feature_dim, split_idx

# --- 2. Train TGNN (for reference, not used in app) ---
def train_tgnn(dataset, feature_dim, num_stocks, epochs=50, lr=1e-4, window_size=30):
    """
    Train TGNN++ model (for reference only, not used in app).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGNNPP(
        num_nodes=num_stocks,
        feature_dim=feature_dim,
        window_size=window_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for node_feats, edge_index, target in train_data:
            node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
            edge_index = edge_index.to(device)
            target = torch.tensor([target], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            pred = model(node_feats, edge_index)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_data):.6f}")

    torch.save(model.state_dict(), os.path.join(Config.MODELS_DIR, "tgnnpp_finbert_model_best.pth"))
    print("TGNN++ model trained and saved.")
    return model

# --- 3. Load TGNN Model ---
def load_tgnn_model(num_stocks, feature_dim, window_size=30, model_path=None):
    """
    Load TGNN++ model from disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGNNPP(
        num_nodes=num_stocks,
        feature_dim=feature_dim,
        window_size=window_size
    ).to(device)
    if model_path is None:
        model_path = os.path.join(Config.MODELS_DIR, "tgnnpp_finbert_model_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# --- 4. Predict with TGNN ---
def predict_tgnn(dataset, model, device, split_idx, mode="test"):
    """
    Predict with TGNN++ model.
    mode: "test" for test set, "inference" for next-day prediction
    Returns: predictions, targets, dates (for test), or (pred_price, last_date) for inference
    """
    if mode == "test":
        test_data = dataset[split_idx:]
        test_preds, test_targets = [], []
        with torch.no_grad():
            for node_feats, edge_index, target in test_data:
                node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
                edge_index = edge_index.to(device)
                pred = model(node_feats, edge_index)
                test_preds.append(pred.cpu().item())
                test_targets.append(target)
        return test_preds, test_targets
    elif mode == "inference":
        # Use last window for inference
        node_feats, edge_index, target = dataset[-1]
        with torch.no_grad():
            node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
            edge_index = edge_index.to(device)
            pred = model(node_feats, edge_index)
            pred_price = pred.cpu().item()
        return pred_price

# --- Helper: Sentiment Embedding ---
def sentiment_df_to_news_embeddings(sentiment_df, embedding_dim=3):
    sentiment_cols = [col for col in sentiment_df.columns if '_sentiment_' in col]
    stock_codes = sorted(set(col.split('_sentiment_')[0] for col in sentiment_cols))
    num_days = len(sentiment_df)
    news_embeddings = np.zeros((num_days, len(stock_codes), embedding_dim), dtype=np.float32)
    for stock_idx, stock in enumerate(stock_codes):
        pos_col = f"{stock}_sentiment_positive"
        neg_col = f"{stock}_sentiment_negative"
        neu_col = f"{stock}_sentiment_neutral"
        vectors = sentiment_df[[pos_col, neg_col, neu_col]].values
        news_embeddings[:, stock_idx, :] = vectors
    return news_embeddings

# --- Helper: Dataset Creation ---
def create_tgnn_dataset(stock_df, macro_df, news_embeddings, target_stock):
    stock_codes = list({col.split('_')[0] for col in stock_df.columns if '_stock' in col})
    num_stocks = len(stock_codes)
    num_days = len(stock_df)
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
    # 3. Add news embeddings (broadcast if needed)
    if news_embeddings.ndim == 2:
        news_embeddings_expanded = np.repeat(news_embeddings[:, np.newaxis, :], num_stocks, axis=1)
    else:
        news_embeddings_expanded = news_embeddings
    feature_tensor = np.concatenate([feature_tensor, news_embeddings_expanded], axis=-1)
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
            prices = np.where(prices <= 0, 1e-6, prices)
            log_returns = np.diff(np.log(prices))
            returns.append(log_returns)
        corr_matrix = np.corrcoef(returns)
        adj_matrix = (corr_matrix > 0.5).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        edge_index = torch.tensor(np.where(adj_matrix), dtype=torch.long)
        node_features = feature_tensor[day_idx-window_size:day_idx]
        target = stock_df[f"{target_stock}_close_stock"].iloc[day_idx]
        dataset.append((node_features, edge_index, target))
    return dataset, stock_codes, feature_tensor.shape[-1]

# --- Example Usage ---
if __name__ == "__main__":
    # Load processed data
    daily_data = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, "final_daily_data.csv"))
    # Prepare data
    dataset, stock_codes, feature_dim, split_idx = prepare_tgnn_data(daily_data, target_stock="AAPL", embedding_dim=3)
    # Load model (always use pre-trained best model for prediction)
    model_path = os.path.join(Config.MODELS_DIR, "tgnnpp_finbert_model_best.pth")
    model, device = load_tgnn_model(num_stocks=len(stock_codes), feature_dim=feature_dim, window_size=30, model_path=model_path)
    # Predict on test set
    test_preds, test_targets = predict_tgnn(dataset, model, device, split_idx, mode="test")
    print(f"Test MSE: {np.mean((np.array(test_preds) - np.array(test_targets))**2):.6f}")
    # Predict next-day close price (inference)
    pred_price = predict_tgnn(dataset, model, device, split_idx, mode="inference")
    print(f"TGNN predicted next-day close price for AAPL: {pred_price:.2f}")