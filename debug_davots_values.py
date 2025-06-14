"""
Debug script to check DAVOTS attribution values in detail
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.data_integration import MultiModalDataIntegrator
from service.xai_analysis import DAVOTS
from model.tgnn_plus_plus import TGNNPlusPlus

def debug_davots_values():
    print("=== DAVOTS Attribution Values Debug ===")
    
    # Load integrated data
    integration_service = MultiModalDataIntegrator()
    ticker = "AAPL"
    date = "2025-06-14"
    
    print(f"Loading integrated data for {ticker} on {date}...")
    integrated_data = integration_service.integrate_all_data(ticker, date)
    
    # Convert to features and sample
    features_df = pd.DataFrame(integrated_data['features'])
    feature_names = list(features_df.columns)
    
    print(f"Total features: {len(feature_names)}")
    print(f"News embedding features: {len([f for f in feature_names if f.startswith('news_emb_')])}")
    
    # Get last few days of data
    features_tensor = torch.tensor(features_df.values[-5:], dtype=torch.float32).unsqueeze(0)
    print(f"Input tensor shape: {features_tensor.shape}")
    
    # Load model
    model_path = "data/processed/model_checkpoint/tgnn_model_AAPL_50epochs.pth"
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    print("Model loaded successfully")
    
    # Create DAVOTS analyzer
    davots_analyzer = DAVOTS(model)
    
    # Test model prediction first
    with torch.no_grad():
        prediction = model(features_tensor)
        print(f"Model prediction: {prediction.item()}")
    
    # Compute raw attributions
    print("\n=== Computing Raw Attributions ===")
    raw_attributions = davots_analyzer.compute_attributions(features_tensor)
    print(f"Raw attributions shape: {raw_attributions.shape}")
    print(f"Raw attributions range: [{raw_attributions.min().item():.2e}, {raw_attributions.max().item():.2e}]")
    print(f"Raw attributions mean: {raw_attributions.mean().item():.2e}")
    print(f"Raw attributions std: {raw_attributions.std().item():.2e}")
    
    # Check non-zero attributions
    non_zero_mask = torch.abs(raw_attributions) > 1e-10
    non_zero_count = non_zero_mask.sum().item()
    print(f"Non-zero attributions (>1e-10): {non_zero_count}/{raw_attributions.numel()}")
    
    # Compute temporal attributions (processed)
    print("\n=== Computing Temporal Attributions ===")
    attribution_results = davots_analyzer.compute_temporal_attributions(
        features_tensor, feature_names, method='integrated_gradients'
    )
    
    attr_matrix = attribution_results['attribution_matrix']
    feature_importance = attribution_results['feature_importance']
    
    print(f"Attribution matrix shape: {attr_matrix.shape}")
    print(f"Feature importance range: [{feature_importance.min():.2e}, {feature_importance.max():.2e}]")
    print(f"Feature importance mean: {feature_importance.mean():.2e}")
    
    # Analyze news embedding attributions specifically
    print("\n=== News Embedding Analysis ===")
    news_indices = [i for i, name in enumerate(feature_names) if name.startswith('news_emb_')]
    news_importance = feature_importance[news_indices]
    
    print(f"News embedding importance range: [{news_importance.min():.2e}, {news_importance.max():.2e}]")
    print(f"News embedding importance mean: {news_importance.mean():.2e}")
    print(f"News embedding importance std: {news_importance.std():.2e}")
    
    # Top 10 news embeddings by importance
    news_feature_scores = [(news_indices[i], feature_names[news_indices[i]], news_importance[i]) 
                          for i in range(len(news_indices))]
    news_feature_scores.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop 10 news embeddings by attribution importance:")
    for i, (idx, name, score) in enumerate(news_feature_scores[:10]):
        print(f"  {i+1:2d}. {name}: {score:.2e}")
    
    # Compare with actual data values
    print("\n=== Comparing with Data Values ===")
    sample_data = features_df.iloc[-1]  # Last day
    for i, (idx, name, score) in enumerate(news_feature_scores[:5]):
        data_value = sample_data[name]
        print(f"  {name}: data={data_value:.4f}, attribution={score:.2e}")
    
    # Check if model gradients are working
    print("\n=== Gradient Check ===")
    features_tensor.requires_grad_(True)
    output = model(features_tensor)
    output.backward()
    
    gradients = features_tensor.grad
    print(f"Gradients shape: {gradients.shape}")
    print(f"Gradients range: [{gradients.min().item():.2e}, {gradients.max().item():.2e}]")
    print(f"Gradients mean: {gradients.mean().item():.2e}")
    print(f"Non-zero gradients: {(torch.abs(gradients) > 1e-10).sum().item()}/{gradients.numel()}")

if __name__ == "__main__":
    debug_davots_values()
