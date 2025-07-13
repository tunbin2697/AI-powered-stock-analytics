import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import joblib

# -------------------------------------
# 1. T-GNN++ MODEL ARCHITECTURE
# -------------------------------------
class TGNNPP(nn.Module):
    """
    T-GNN++ Transformer Architecture for Multi-Modal Financial Forecasting.
    This model predicts the next-day price difference.
    """
    def __init__(self, num_nodes, feature_dim, window_size=30, embed_dim=128, gat_heads=2, trans_heads=8, trans_layers=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        gat_out_dim = 64

        # A. Feature Embedding Layer: Projects combined features into a dense space
        self.feature_embed = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

        # B. T-GNN Module: Captures spatial (inter-stock) correlations at each timestep
        self.gat = GATConv(embed_dim, gat_out_dim, heads=gat_heads, concat=True)
        
        # C. Transformer Module: Captures temporal patterns across the window
        transformer_input_dim = gat_out_dim * gat_heads
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_input_dim, nhead=trans_heads, batch_first=True),
            num_layers=trans_layers
        )

        # D. Prediction Head: Final layers to predict the price difference
        self.pred_head = nn.Sequential(
            nn.Linear(transformer_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        # Hook to capture transformer attention
        self.transformer_attention_weights = None
        self.transformer.layers[-1].self_attn.register_forward_hook(self._save_transformer_attention)

    def _save_transformer_attention(self, module, input, output):
        # output[1] contains the attention weights from the MultiheadAttention layer
        self.transformer_attention_weights = output[1]

    def forward(self, node_feats, edge_indices, return_attention=False):
        # node_feats: [batch_size, window_size, num_nodes, feature_dim]
        batch_size, window_size, num_nodes, _ = node_feats.shape

        # 1. Embed features
        embedded = self.feature_embed(node_feats.view(-1, node_feats.shape[-1]))
        embedded = embedded.view(batch_size, window_size, num_nodes, -1)

        # 2. Process each time step with GAT
        gat_outputs = []
        all_gat_attentions = []
        for t in range(window_size):
            x_t = embedded[:, t, :, :].reshape(batch_size * num_nodes, -1)
            
            batch_edge_index = edge_indices[0].clone().to(node_feats.device)
            if batch_size > 1:
                for i in range(1, batch_size):
                    batch_edge_index = torch.cat([batch_edge_index, edge_indices[i] + i * num_nodes], dim=1)

            # Modify GAT call to return attention weights
            gat_out, gat_attention = self.gat(x_t, batch_edge_index, return_attention_weights=True)
            
            if return_attention:
                all_gat_attentions.append(gat_attention)

            gat_out = gat_out.view(batch_size, num_nodes, -1)
            gat_outputs.append(gat_out.unsqueeze(1))

        # 3. Aggregate spatial features over time
        temporal_data = torch.cat(gat_outputs, dim=1)
        transformer_input = temporal_data.mean(dim=2)

        # 4. Transformer processing for temporal patterns
        transformer_output = self.transformer(transformer_input)
        final_representation = transformer_output[:, -1, :]

        # 5. Predict the price difference
        prediction = self.pred_head(final_representation).squeeze(-1)

        if return_attention:
            return prediction, all_gat_attentions, self.transformer_attention_weights
        
        return prediction

# -------------------------------------
# 2. DATA PREPARATION FUNCTIONS
# -------------------------------------

def generate_finbert_embeddings(news_df, batch_size=16):
    """Processes news titles into 768D FinBERT embeddings."""
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertModel.from_pretrained('ProsusAI/finbert')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = model.eval()

    stock_codes = [col.split('_')[0] for col in news_df.columns if '_title' in col]
    news_df = news_df.reset_index(drop=True)
    num_days = len(news_df)
    embeddings = np.zeros((num_days, len(stock_codes), 768), dtype=np.float32)

    for i in tqdm(range(num_days), desc="Processing FinBERT Embeddings"):
        texts_for_day = [str(news_df.at[i, f"{code}_title"]) for code in stock_codes]
        
        inputs = tokenizer(texts_for_day, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            day_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # CLS token
        
        embeddings[i] = day_embeddings
        
    return embeddings

def scale_features(stock_df, macro_df):
    """Scales stock and macroeconomic data using MinMaxScaler."""
    stock_features = stock_df.drop(columns=['Date'])
    macro_features = macro_df.drop(columns=['Date'])

    stock_scaler = MinMaxScaler()
    scaled_stock_data = stock_scaler.fit_transform(stock_features)
    scaled_stock_df = pd.DataFrame(scaled_stock_data, columns=stock_features.columns, index=stock_df.index)
    
    macro_scaler = MinMaxScaler()
    scaled_macro_data = macro_scaler.fit_transform(macro_features)
    scaled_macro_df = pd.DataFrame(scaled_macro_data, columns=macro_features.columns, index=macro_df.index)
    
    # Add date back for alignment
    scaled_stock_df['Date'] = stock_df['Date'].values
    scaled_macro_df['Date'] = macro_df['Date'].values
    
    return scaled_stock_df, scaled_macro_df, stock_scaler, macro_scaler

# -------------------------------------
# 3. DATASET CREATION FUNCTION
# -------------------------------------

def create_tgnn_dataset(unscaled_stock_df, scaled_stock_df, scaled_macro_df, news_embeddings, target_stock, window_size=30, corr_threshold=0.5):
    """
    Combines all data sources and creates windowed samples for T-GNN++.
    The target is the next day's price difference.
    """
    stock_codes = sorted(list({col.split('_')[0] for col in scaled_stock_df.columns if '_stock' in col}))
    num_stocks = len(stock_codes)
    num_days = len(scaled_stock_df)

    # 1. Combine all SCALED features into a single tensor
    # Note: We use scaled data for features, but unscaled for target calculation
    stock_features = np.stack([scaled_stock_df[[col for col in scaled_stock_df.columns if col.startswith(code)]].values for code in stock_codes], axis=1)
    macro_features = np.repeat(scaled_macro_df.drop(columns=['Date']).values[:, np.newaxis, :], num_stocks, axis=1)
    feature_tensor = np.concatenate([stock_features, macro_features, news_embeddings], axis=-1)

    # 2. Create windowed dataset
    dataset = []
    
    for day_idx in range(window_size, num_days - 1): # Ensure there's a next day for the target
        # a. Dynamic graph construction from rolling window returns
        window_returns = []
        for code in stock_codes:
            # Use unscaled close prices for accurate correlation
            close_prices = unscaled_stock_df[f"{code}_close_stock"].iloc[day_idx-window_size:day_idx].values
            log_returns = np.diff(np.log(close_prices + 1e-9))
            window_returns.append(log_returns)
        
        corr_matrix = np.corrcoef(window_returns)
        adj_matrix = (np.abs(corr_matrix) > corr_threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        edge_index = torch.tensor(np.where(adj_matrix), dtype=torch.long)

        # b. Node features for the window (from the combined, scaled tensor)
        node_features = feature_tensor[day_idx-window_size:day_idx]

        # c. Target: Next day's price difference (from unscaled data)
        current_price = unscaled_stock_df[f"{target_stock}_close_stock"].iloc[day_idx - 1]
        next_day_price = unscaled_stock_df[f"{target_stock}_close_stock"].iloc[day_idx]
        target_diff = next_day_price - current_price
        
        # d. Metadata for evaluation and price reconstruction
        meta_info = {'current_price': current_price, 'actual_next_price': next_day_price}

        dataset.append((node_features, edge_index, target_diff, meta_info))

    return dataset, stock_codes, feature_tensor.shape[-1]

# -------------------------------------
# 4. TRAINING AND EVALUATION FUNCTIONS
# -------------------------------------

def train_tgnn_model(dataset, feature_dim, num_stocks, model_save_path, epochs=20, lr=1e-4, batch_size=16):
    """Trains the T-GNN++ model and saves the best version."""
    split_idx = int(0.8 * len(dataset))
    train_data, val_data = dataset[:split_idx], dataset[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGNNPP(num_nodes=num_stocks, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs} Training"):
            batch = train_data[i:i+batch_size]
            if not batch: continue
            
            node_feats, edge_indices, targets, _ = zip(*batch)
            
            node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32).to(device)
            targets = torch.tensor(np.array(targets), dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            pred_diffs = model(node_feats, edge_indices)
            loss = criterion(pred_diffs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for node_feats, edge_index, target_diff, _ in val_data:
                node_feats = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
                target_diff = torch.tensor([target_diff], dtype=torch.float32).to(device)
                pred_diff = model(node_feats, [edge_index])
                val_loss = criterion(pred_diff, target_diff)
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_data)
        avg_val_loss = total_val_loss / len(val_data)
        print(f"Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with validation loss {best_val_loss:.6f}")

def test_and_plot_tgnn(test_dataset, feature_dim, num_stocks, model_path):
    """Loads a trained T-GNN++ model, evaluates it, and plots the results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGNNPP(num_nodes=num_stocks, feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    reconstructed_prices, actual_prices = [], []

    with torch.no_grad():
        for node_feats, edge_index, _, meta_info in tqdm(test_dataset, desc="Testing Model"):
            node_feats_tensor = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred_diff = model(node_feats_tensor, [edge_index]).cpu().item()
            
            # Reconstruct the predicted price for plotting
            reconstructed_price = meta_info['current_price'] + pred_diff
            reconstructed_prices.append(reconstructed_price)
            actual_prices.append(meta_info['actual_next_price'])

    # Calculate Metrics on reconstructed prices
    mse_price = np.mean((np.array(reconstructed_prices) - np.array(actual_prices))**2)
    rmse_price = np.sqrt(mse_price)
    print(f"\n--- Test Results ---")
    print(f"MSE on Reconstructed Price: {mse_price:.4f}")
    print(f"RMSE on Reconstructed Price: {rmse_price:.4f}")

    # Plotting Actual vs. Predicted Close Price
    plt.figure(figsize=(16, 8))
    plt.plot(actual_prices, label='Actual Close Price', color='navy', marker='.', alpha=0.7)
    plt.plot(reconstructed_prices, label='Predicted Close Price', color='orangered', linestyle='--', marker='x', alpha=0.7)
    plt.title("T-GNN++: Actual vs. Predicted Close Price (Reconstructed from Difference)", fontsize=16)
    plt.xlabel("Test Set Timesteps", fontsize=12)
    plt.ylabel("Stock Price (USD)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------------
# 6. XAI VISUALIZATION FUNCTIONS
# -------------------------------------

def visualize_icfts_gat_attention(model, data_sample, stock_codes, time_step_to_viz=-1):
    """
    Visualizes GAT attention scores for a specific time step (ICFTS).
    This shows inter-stock influence.
    """
    model.eval()
    device = next(model.parameters()).device
    node_feats, edge_index, _, _ = data_sample
    
    node_feats_tensor = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, gat_attentions, _ = model(node_feats_tensor, [edge_index], return_attention=True)

    # We focus on the attention at a specific time step (e.g., the last one)
    attention_at_t = gat_attentions[time_step_to_viz]
    
    # The attention_weights tensor shape is (num_edges, num_heads)
    # We average over the heads for a single score per edge
    avg_attention = attention_at_t[1].mean(dim=1).cpu().numpy()
    
    # Create an adjacency matrix for visualization
    num_nodes = len(stock_codes)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    edges = attention_at_t[0].cpu().numpy()
    adj_matrix[edges[0], edges[1]] = avg_attention
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(adj_matrix, cmap='viridis')
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(stock_codes)))
    ax.set_yticks(np.arange(len(stock_codes)))
    ax.set_xticklabels(stock_codes, rotation=90)
    ax.set_yticklabels(stock_codes)
    
    ax.set_xlabel("Source Stock (Influencer)")
    ax.set_ylabel("Target Stock (Influenced)")
    ax.set_title(f"ICFTS: GAT Attention Heatmap (Inter-Stock Influence) at Time Step {time_step_to_viz}")
    plt.tight_layout()
    plt.show()


def visualize_davots_transformer_attention(model, data_sample, window_size):
    """
    Visualizes Transformer self-attention scores (DAVOTS).
    This shows the temporal importance of past days.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # --- FIX: Re-register the hook on the loaded model ---
    # This ensures the attention weights are captured even after loading from a file.
    if not hasattr(model, 'transformer_attention_weights'):
        model.transformer_attention_weights = None
        model.transformer.layers[-1].self_attn.register_forward_hook(model._save_transformer_attention)
    
    node_feats, edge_index, _, _ = data_sample
    node_feats_tensor = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # The hook will now correctly store the attention weights
        _ = model(node_feats_tensor, [edge_index], return_attention=True)
    
    # Attention weights shape: (batch_size, seq_len, seq_len)
    # We are interested in the attention given by the last time step to all previous steps
    attention_weights = model.transformer_attention_weights[0, -1, :].cpu().numpy()
    
    # Plotting
    plt.figure(figsize=(14, 6))
    plt.bar(range(window_size), attention_weights, color='skyblue')
    plt.xlabel("Time Steps (Days before prediction)")
    plt.ylabel("Attention Score")
    plt.title("DAVOTS: Transformer Self-Attention (Temporal Importance)")
    plt.xticks(np.arange(window_size), [f"t-{window_size-i-1}" for i in range(window_size)], rotation=90)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()


# -------------------------------------
# 7. EXAMPLE USAGE (UPDATED)
# -------------------------------------
if __name__ == '__main__':
    # --- Create Dummy Data (to mimic your setup) ---
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=300))
    stock_data = {
        'Date': dates,
        'AAPL_close_stock': 150 + np.random.randn(300).cumsum(),
        'AAPL_rsi14_stock': 50 + np.random.randn(300) * 10,
        'TSLA_close_stock': 800 + np.random.randn(300).cumsum() * 2,
        'TSLA_rsi14_stock': 50 + np.random.randn(300) * 10,
    }
    unscaled_stock_df = pd.DataFrame(stock_data)

    news_data = {
        'Date': dates,
        'AAPL_title': ["News about Apple"] * 300,
        'TSLA_title': ["News about Tesla"] * 300
    }
    news_df = pd.DataFrame(news_data)

    macro_data = {
        'Date': dates,
        'GDP_macro': 20000 + np.random.randn(300).cumsum() * 10,
        'CPI_macro': 280 + np.random.randn(300).cumsum() * 0.1,
    }
    macro_df = pd.DataFrame(macro_data)

    TARGET_STOCK = 'AAPL'
    MODEL_SAVE_PATH = f'./models/tgnnpp_{TARGET_STOCK}.pth'

    # --- Full Pipeline ---
    # 1. Scale features
    print("Step 1: Scaling features...")
    scaled_stock_df, scaled_macro_df, _, _ = scale_features(unscaled_stock_df, macro_df)
    
    # 2. Generate News Embeddings (or load if already created)
    print("\nStep 2: Generating FinBERT embeddings...")
    news_embeddings = generate_finbert_embeddings(news_df)
    
    # 3. Create the T-GNN++ dataset
    print("\nStep 3: Creating T-GNN++ dataset...")
    dataset, stock_codes, feature_dim = create_tgnn_dataset(
        unscaled_stock_df, 
        scaled_stock_df, 
        scaled_macro_df, 
        news_embeddings, 
        target_stock=TARGET_STOCK
    )
    print(f"Dataset created with {len(dataset)} samples. Feature dimension per node: {feature_dim}")

    # 4. Train the model
    print("\nStep 4: Training T-GNN++ model...")
    train_tgnn_model(
        dataset, 
        feature_dim=feature_dim, 
        num_stocks=len(stock_codes),
        model_save_path=MODEL_SAVE_PATH,
        epochs=5 # Using fewer epochs for quick demonstration
    )

    # 5. Test the model and plot results
    print("\nStep 5: Testing model and plotting results...")
    test_split_index = int(0.8 * len(dataset))
    test_dataset = dataset[test_split_index:]
    test_and_plot_tgnn(test_dataset, feature_dim, len(stock_codes), MODEL_SAVE_PATH)

    # 6. Generate XAI Visualizations for a sample from the test set
    print("\nStep 6: Generating XAI visualizations for one test sample...")
    xai_sample = test_dataset[0] # Use the first sample of the test set for explanation
    
    # Load the trained model
    model = TGNNPP(num_nodes=len(stock_codes), feature_dim=feature_dim)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # Move model to device
    
    # ICFTS Plot
    visualize_icfts_gat_attention(model, xai_sample, stock_codes)
    
    # DAVOTS Plot
    visualize_davots_transformer_attention(model, xai_sample, window_size=30)

    print("\nPipeline finished.")