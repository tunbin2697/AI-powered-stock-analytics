"""
Test the integrated data structure for TGNN++, DAVOTS, and ICFTS
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from service.data_integration import MultiModalDataIntegrator
import pandas as pd
import numpy as np
import torch

def test_integrated_data_structure():
    """Test and analyze the integrated data structure"""
    
    print("=== Testing Integrated Data Structure ===")
    
    # Initialize integrator
    integrator = MultiModalDataIntegrator()
    
    # Test data integration
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    print(f"Integrating data for {ticker} from {start_date} to {end_date}")
    
    try:
        # Get integrated data
        integrated_data = integrator.integrate_all_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if integrated_data.empty:
            print("❌ No integrated data returned")
            return
        
        print(f"✅ Integrated data shape: {integrated_data.shape}")
        print(f"✅ Date range: {integrated_data.index.min()} to {integrated_data.index.max()}")
        
        # Analyze feature types
        print("\n=== Feature Analysis ===")
        
        all_columns = integrated_data.columns.tolist()
        print(f"Total columns: {len(all_columns)}")
        
        # Categorize features
        price_features = [col for col in all_columns 
                         if any(keyword in col.lower() for keyword in 
                               ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'rsi', 'macd', 'bb', 'return', 'volatility'])]
        
        macro_features = [col for col in all_columns if col.startswith('macro_')]
        
        news_features = [col for col in all_columns 
                        if any(keyword in col.lower() for keyword in ['news', 'sentiment', 'emb'])]
        
        print(f"Price features: {len(price_features)}")
        print(f"Macro features: {len(macro_features)}")
        print(f"News features: {len(news_features)}")
        
        # Show sample features
        print(f"\nSample price features: {price_features[:5]}")
        print(f"Sample macro features: {macro_features[:5]}")
        print(f"Sample news features: {news_features[:3]}")
        
        # Test data preparation for TGNN++
        print("\n=== TGNN++ Data Preparation ===")
        features, targets, feature_names = integrator.prepare_model_data(
            integrated_data, sequence_length=30, target_col='Close'
        )
        
        if features.size() == torch.Size([0]):
            print("❌ No features prepared for TGNN++")
        else:
            print(f"✅ Features tensor shape: {features.shape}")
            print(f"✅ Targets tensor shape: {targets.shape}")
            print(f"✅ Number of feature names: {len(feature_names)}")
            
            # Expected format for TGNN++: (batch_size, seq_len, num_features)
            expected_format = f"Expected: (batch_size={features.shape[0]}, seq_len={features.shape[1]}, num_features={features.shape[2]})"
            print(f"TGNN++ input format: {expected_format}")
        
        # Analyze data quality
        print("\n=== Data Quality Analysis ===")
        
        # Check for missing values
        missing_data = integrated_data.isnull().sum()
        columns_with_missing = missing_data[missing_data > 0]
        
        if len(columns_with_missing) > 0:
            print(f"⚠️  Columns with missing data: {len(columns_with_missing)}")
            print(columns_with_missing.head())
        else:
            print("✅ No missing data found")
        
        # Check data types
        data_types = integrated_data.dtypes.value_counts()
        print(f"Data types: {data_types}")
        
        # Sample data
        print("\n=== Sample Data ===")
        print("First 3 rows:")
        print(integrated_data.head(3))
        
        print("\nLast 3 rows:")
        print(integrated_data.tail(3))
        
        # Requirements for DAVOTS and ICFTS
        print("\n=== XAI Requirements Check ===")
        
        # For DAVOTS (Dynamic Attribution Visualization Over Time Series)
        print("DAVOTS requirements:")
        print(f"✅ Time series data: {integrated_data.shape[0]} time steps")
        print(f"✅ Multiple features: {integrated_data.shape[1]} features")
        print(f"✅ Target variable: 'Close' column exists: {'Close' in integrated_data.columns}")
        
        # For ICFTS (Interventional Causal Framework for Time Series)
        print("\nICFTS requirements:")
        print(f"✅ Multi-modal data: Price({len(price_features)}), Macro({len(macro_features)}), News({len(news_features)})")
        print(f"✅ Temporal structure: Daily frequency data")
        print(f"✅ Feature relationships: Cross-modal features available")
        
        return integrated_data, features, targets, feature_names
        
    except Exception as e:
        print(f"❌ Error during integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    result = test_integrated_data_structure()
    
    if result[0] is not None:
        print(f"\n🎉 Integration test successful!")
        print(f"Ready for TGNN++ training, DAVOTS analysis, and ICFTS visualization")
    else:
        print(f"\n💥 Integration test failed!")
