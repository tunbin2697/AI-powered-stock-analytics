"""
TGNN++ (Temporal Graph Neural Network Plus Plus) implementation.
Enhanced version with attention mechanism and multi-modal feature handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for capturing time-dependent relationships."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(TemporalAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
        Returns:
            Attention-weighted features and attention weights
        """
        batch_size, seq_len, features = x.shape
        
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        return attended, attention_weights

class MultiModalFusion(nn.Module):
    """Multi-modal fusion layer for combining different data types."""
    
    def __init__(self, price_dim: int, macro_dim: int, news_dim: int, output_dim: int):
        super(MultiModalFusion, self).__init__()
        self.price_dim = price_dim
        self.macro_dim = macro_dim
        self.news_dim = news_dim
        self.output_dim = output_dim
        
        # Calculate projection dimensions to ensure they sum to output_dim
        proj_dim = output_dim // 3
        remaining_dim = output_dim - (proj_dim * 2)  # Handle division remainder
        
        # Individual projections
        self.price_proj = nn.Linear(price_dim, proj_dim)
        self.macro_proj = nn.Linear(macro_dim, proj_dim)
        self.news_proj = nn.Linear(news_dim, remaining_dim)  # Use remaining dimension
        
        # Fusion layer
        self.fusion = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, price_features, macro_features, news_features):
        """Fuse multi-modal features."""
        price_proj = self.price_proj(price_features)
        macro_proj = self.macro_proj(macro_features)
        news_proj = self.news_proj(news_features)
        
        # Concatenate and fuse
        combined = torch.cat([price_proj, macro_proj, news_proj], dim=-1)
        fused = self.fusion(combined)
        fused = self.dropout(fused)
        
        return fused

class GraphConvolution(nn.Module):
    """Graph convolution layer for capturing feature relationships."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj_matrix):
        """
        Args:
            x: Node features (batch_size, seq_len, num_nodes, features)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)
        
        # Graph convolution
        output = torch.matmul(adj_matrix, support) + self.bias
        
        return F.relu(output)

class TGNNPlusPlus(nn.Module):
    """
    TGNN++ model for multi-modal time series prediction with XAI support.
    Combines temporal attention, graph convolution, and multi-modal fusion.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super(TGNNPlusPlus, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Feature categorization (approximate split)
        self.price_dim = min(23, input_dim)  # Price/volume features
        self.macro_dim = min(19, max(0, input_dim - self.price_dim - 768))  # Macro features
        self.news_dim = min(768, max(0, input_dim - self.price_dim - self.macro_dim))  # News embeddings
        
        logger.info(f"Feature dimensions - Price: {self.price_dim}, Macro: {self.macro_dim}, News: {self.news_dim}")
        
        # Multi-modal fusion
        if self.macro_dim > 0 and self.news_dim > 0:
            self.fusion = MultiModalFusion(
                self.price_dim, self.macro_dim, self.news_dim, hidden_dim
            )
            fusion_output_dim = hidden_dim
        else:
            # Fallback: simple linear projection
            self.fusion = nn.Linear(input_dim, hidden_dim)
            fusion_output_dim = hidden_dim
        
        # Temporal attention layers
        self.attention_layers = nn.ModuleList([
            TemporalAttention(fusion_output_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
          # Graph structure (learned adjacency matrix)
        self.num_nodes = min(10, hidden_dim // 16)  # Ensure divisibility
        if self.num_nodes == 0:
            self.num_nodes = 1
        
        # Make sure hidden_dim is divisible by num_nodes
        node_feature_dim = hidden_dim // self.num_nodes
        if hidden_dim % self.num_nodes != 0:
            # Adjust to make it divisible
            node_feature_dim = hidden_dim // self.num_nodes + 1
            self.num_nodes = hidden_dim // node_feature_dim
            if self.num_nodes == 0:
                self.num_nodes = 1
                node_feature_dim = hidden_dim
        
        logger.info(f"Graph structure - Nodes: {self.num_nodes}, Features per node: {node_feature_dim}")
        
        self.adjacency = nn.Parameter(torch.randn(self.num_nodes, self.num_nodes))
        
        # Graph convolution layers
        self.graph_conv = GraphConvolution(node_feature_dim, node_feature_dim)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # For XAI: store intermediate activations
        self.attention_weights = []
        self.feature_importance = None
        
    def forward(self, x, return_attention=False):
        """
        Forward pass through TGNN++.
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            return_attention: Whether to return attention weights for XAI
            
        Returns:
            Predictions and optionally attention weights
        """
        batch_size, seq_len, features = x.shape
        
        # Multi-modal fusion
        if hasattr(self.fusion, 'price_proj') and self.macro_dim > 0 and self.news_dim > 0:
            # Split features by modality
            price_features = x[:, :, :self.price_dim]
            macro_features = x[:, :, self.price_dim:self.price_dim + self.macro_dim]
            news_features = x[:, :, -self.news_dim:]
            
            fused = self.fusion(price_features, macro_features, news_features)
        else:
            # Simple fusion for fallback
            fused = self.fusion(x)
        
        # Temporal attention
        self.attention_weights = []
        attended = fused
        
        for attention_layer in self.attention_layers:
            attended, attn_weights = attention_layer(attended)
            if return_attention:
                self.attention_weights.append(attn_weights)
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(attended)
          # Graph convolution (reshape for graph processing)
        if self.num_nodes > 1:
            # Calculate node feature dimension
            node_feature_dim = lstm_output.shape[-1] // self.num_nodes
            
            # Only proceed if dimensions work out
            if lstm_output.shape[-1] % self.num_nodes == 0:
                # Reshape for graph processing
                graph_input = lstm_output.view(batch_size, seq_len, self.num_nodes, node_feature_dim)
                
                # Normalize adjacency matrix
                adj_norm = F.softmax(self.adjacency, dim=1)
                
                # Apply graph convolution
                graph_output = self.graph_conv(graph_input, adj_norm)
                
                # Reshape back
                graph_output = graph_output.view(batch_size, seq_len, -1)
            else:
                # Skip graph convolution if dimensions don't match
                logger.warning(f"Skipping graph convolution due to dimension mismatch: {lstm_output.shape[-1]} not divisible by {self.num_nodes}")
                graph_output = lstm_output
        else:
            graph_output = lstm_output
        
        # Use last timestep for prediction
        final_features = graph_output[:, -1, :]
        
        # Output projection
        output = self.output_projection(final_features)
        
        if return_attention:
            return output, self.attention_weights
        else:
            return output
    
    def get_feature_importance(self, x):
        """Calculate feature importance for XAI (DAVOTS)."""
        x.requires_grad_(True)
        
        # Forward pass
        output = self.forward(x)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Feature importance as gradient magnitude
        importance = torch.abs(gradients).mean(dim=0)  # Average over batch
        
        return importance.detach()

class TGNNTrainer:
    """Trainer class for TGNN++ model."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_x)
            loss = self.criterion(predictions.squeeze(), batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions.squeeze(), batch_y)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'model/tgnn_plus_plus_best.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('model/tgnn_plus_plus_best.pth'))
        
        return train_losses, val_losses

# Example usage
def create_model_from_data(input_dim: int):
    """Create TGNN++ model with appropriate dimensions."""
    model = TGNNPlusPlus(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        output_dim=1
    )
    
    logger.info(f"Created TGNN++ model with input_dim={input_dim}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    test_input_dim = 810  # Example: 23 price + 19 macro + 768 news
    model = create_model_from_data(test_input_dim)
    
    # Test forward pass
    batch_size, seq_len = 32, 30
    test_input = torch.randn(batch_size, seq_len, test_input_dim)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Model output shape: {output.shape}")
        
        # Test attention return
        output_with_attn, attention_weights = model(test_input, return_attention=True)
        print(f"Number of attention layers: {len(attention_weights)}")
        print(f"Attention weights shape: {attention_weights[0].shape}")
