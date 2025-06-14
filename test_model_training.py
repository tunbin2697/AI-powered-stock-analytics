"""
Comprehensive test script to train the TGNN++ model.
Tests the complete pipeline: data integration -> sequence preparation -> model training -> evaluation.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from service.data_integration import MultiModalDataIntegrator
from model.tgnn_plus_plus import TGNNPlusPlus, TGNNTrainer
from service.xai_analysis import XAIAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TGNNModelTrainingTest:
    """Comprehensive test for TGNN++ model training."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Configuration
        self.config = {
            'symbol': 'AAPL',
            'date_suffix': '2025-06-14',
            'sequence_length': 30,
            'batch_size': 16,  # Smaller batch size for testing
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'hidden_dim': 64,  # Smaller for testing
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 10,  # Fewer epochs for testing
            'early_stopping_patience': 5,
            'target_column': 'Close'
        }
        
        # Initialize components
        self.data_integrator = None
        self.model = None
        self.trainer = None
        self.integrated_data = None
        self.feature_names = None
        
        # Create directories
        os.makedirs('model', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def step1_integrate_data(self):
        """Step 1: Integrate multi-modal data."""
        logger.info("🔄 Step 1: Integrating multi-modal data...")
        
        self.data_integrator = MultiModalDataIntegrator()
        
        # Test data integration
        integrated_data = self.data_integrator.integrate_all_data(
            symbol=self.config['symbol'],
            date_suffix=self.config['date_suffix']
        )
        
        if integrated_data.empty:
            raise ValueError("❌ Data integration failed - no data returned")
        
        self.integrated_data = integrated_data
        self.feature_names = self.data_integrator.get_feature_names()
        
        logger.info(f"✅ Data integration successful!")
        logger.info(f"   - Dataset shape: {integrated_data.shape}")
        logger.info(f"   - Features: {len(self.feature_names)}")
        logger.info(f"   - Date range: {integrated_data.index.min()} to {integrated_data.index.max()}")
        
        # Analyze feature composition
        price_features = [f for f in self.feature_names if not f.startswith('macro_') and not f.startswith('news_')]
        macro_features = [f for f in self.feature_names if f.startswith('macro_')]
        news_features = [f for f in self.feature_names if f.startswith('news_')]
        
        logger.info(f"   - Price features: {len(price_features)}")
        logger.info(f"   - Macro features: {len(macro_features)}")
        logger.info(f"   - News features: {len(news_features)}")
        
        # Save integrated data
        integrated_data.to_csv('data/processed/test_integrated_data.csv')
        logger.info("   - Saved integrated data to CSV")
        
        return {
            'total_features': len(self.feature_names),
            'price_features': len(price_features),
            'macro_features': len(macro_features),
            'news_features': len(news_features),
            'data_shape': integrated_data.shape
        }
    
    def step2_prepare_sequences(self, data_info):
        """Step 2: Create sequences for training."""
        logger.info("🔄 Step 2: Preparing sequences for training...")
        
        if self.integrated_data is None:
            raise ValueError("❌ No integrated data available")
        
        # Prepare features and target
        target_col = self.config['target_column']
        if target_col not in self.integrated_data.columns:
            raise ValueError(f"❌ Target column '{target_col}' not found in data")
        
        # Get features (excluding target)
        feature_data = self.integrated_data[self.feature_names].values
        target_data = self.integrated_data[target_col].values
        
        logger.info(f"   - Feature data shape: {feature_data.shape}")
        logger.info(f"   - Target data shape: {target_data.shape}")
        
        # Create sequences
        sequence_length = self.config['sequence_length']
        X, y = [], []
        
        for i in range(len(feature_data) - sequence_length):
            X.append(feature_data[i:i+sequence_length])
            y.append(target_data[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"   - Created {len(X)} sequences")
        logger.info(f"   - X shape: {X.shape}")
        logger.info(f"   - y shape: {y.shape}")
        
        # Split data
        train_size = int(len(X) * self.config['train_split'])
        val_size = int(len(X) * self.config['val_split'])
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"   - Train: {X_train.shape[0]} samples")
        logger.info(f"   - Validation: {X_val.shape[0]} samples")
        logger.info(f"   - Test: {X_test.shape[0]} samples")
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        logger.info("✅ Sequence preparation successful!")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'X_test': X_test,
            'y_test': y_test,
            'input_dim': X_train.shape[2]
        }
    
    def step3_create_model(self, input_dim):
        """Step 3: Create and initialize TGNN++ model."""
        logger.info("🔄 Step 3: Creating TGNN++ model...")
        
        self.model = TGNNPlusPlus(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            output_dim=1
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model created successfully!")
        logger.info(f"   - Input dimension: {input_dim}")
        logger.info(f"   - Hidden dimension: {self.config['hidden_dim']}")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def step4_train_model(self, train_loader, val_loader):
        """Step 4: Train the TGNN++ model."""
        logger.info("🔄 Step 4: Training TGNN++ model...")
        
        if self.model is None:
            raise ValueError("❌ Model not created yet")
        
        # Create trainer
        self.trainer = TGNNTrainer(self.model, device=self.device)
        
        # Train model
        train_losses, val_losses = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['epochs'],
            early_stopping_patience=self.config['early_stopping_patience']
        )
        
        logger.info("✅ Model training completed!")
        logger.info(f"   - Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"   - Final validation loss: {val_losses[-1]:.4f}")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def step5_evaluate_model(self, test_loader, X_test, y_test):
        """Step 5: Evaluate model performance."""
        logger.info("🔄 Step 5: Evaluating model performance...")
        
        if self.model is None or self.trainer is None:
            raise ValueError("❌ Model not trained yet")
        
        # Evaluate on test set
        test_loss = self.trainer.validate(test_loader)
        
        # Get predictions
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred = self.model(batch_x)
                predictions.extend(pred.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        
        logger.info("✅ Model evaluation completed!")
        logger.info(f"   - Test Loss: {test_loss:.4f}")
        logger.info(f"   - MSE: {mse:.4f}")
        logger.info(f"   - MAE: {mae:.4f}")
        logger.info(f"   - RMSE: {rmse:.4f}")
        logger.info(f"   - R²: {r2:.4f}")
        
        # Plot predictions vs actuals
        self.plot_predictions(actuals, predictions)
        
        return {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def step6_xai_analysis(self, X_test):
        """Step 6: XAI analysis for model interpretability."""
        logger.info("🔄 Step 6: Performing XAI analysis...")
        
        if self.model is None:
            raise ValueError("❌ Model not trained yet")
        
        # Test with a single sample
        test_sample = X_test[:1].to(self.device)  # First test sample
        
        # Get feature importance
        feature_importance = self.model.get_feature_importance(test_sample)
        
        # Get attention weights
        _, attention_weights = self.model(test_sample, return_attention=True)
        
        logger.info("✅ XAI analysis completed!")
        logger.info(f"   - Feature importance shape: {feature_importance.shape}")
        logger.info(f"   - Number of attention layers: {len(attention_weights)}")
        
        # Plot feature importance
        self.plot_feature_importance(feature_importance)
        
        return {
            'feature_importance': feature_importance.cpu().numpy(),
            'attention_weights': [attn.cpu().numpy() for attn in attention_weights]
        }
    
    def plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('TGNN++ Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("   - Saved training curves plot")
    
    def plot_predictions(self, actuals, predictions):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(actuals, predictions, alpha=0.6)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        
        # Time series plot (last 100 points)
        plt.subplot(2, 2, 2)
        n_show = min(100, len(actuals))
        plt.plot(actuals[-n_show:], label='Actual', alpha=0.8)
        plt.plot(predictions[-n_show:], label='Predicted', alpha=0.8)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Time Series (Last {n_show} points)')
        plt.legend()
        
        # Residuals
        plt.subplot(2, 2, 3)
        residuals = actuals - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Error distribution
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        plt.tight_layout()
        plt.savefig('results/predictions_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("   - Saved predictions analysis plot")
    
    def plot_feature_importance(self, feature_importance):
        """Plot feature importance for XAI."""
        if self.feature_names is None:
            logger.warning("   - No feature names available for plotting")
            return
        
        # Get top 20 most important features
        importance_values = feature_importance.mean(dim=0).cpu().numpy()  # Average over sequence length
        
        # Create feature importance dataframe
        df_importance = pd.DataFrame({
            'feature': self.feature_names[:len(importance_values)],
            'importance': importance_values
        })
        
        # Sort by importance
        df_importance = df_importance.sort_values('importance', ascending=False)
        
        # Plot top 20
        plt.figure(figsize=(12, 8))
        top_features = df_importance.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance (DAVOTS)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("   - Saved feature importance plot")
    
    def run_complete_test(self):
        """Run the complete training test pipeline."""
        logger.info("🚀 Starting TGNN++ Model Training Test")
        logger.info("=" * 60)
        
        try:
            # Step 1: Data Integration
            data_info = self.step1_integrate_data()
            
            # Step 2: Sequence Preparation
            sequence_info = self.step2_prepare_sequences(data_info)
            
            # Step 3: Model Creation
            model = self.step3_create_model(sequence_info['input_dim'])
            
            # Step 4: Model Training
            train_losses, val_losses = self.step4_train_model(
                sequence_info['train_loader'], 
                sequence_info['val_loader']
            )
            
            # Step 5: Model Evaluation
            eval_results = self.step5_evaluate_model(
                sequence_info['test_loader'],
                sequence_info['X_test'],
                sequence_info['y_test']
            )
            
            # Step 6: XAI Analysis
            xai_results = self.step6_xai_analysis(sequence_info['X_test'])
            
            logger.info("=" * 60)
            logger.info("🎉 TGNN++ Model Training Test COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            # Summary
            logger.info("📊 FINAL RESULTS SUMMARY:")
            logger.info(f"   - Dataset: {data_info['data_shape']}")
            logger.info(f"   - Features: {data_info['total_features']} (Price: {data_info['price_features']}, Macro: {data_info['macro_features']}, News: {data_info['news_features']})")
            logger.info(f"   - Test R²: {eval_results['r2']:.4f}")
            logger.info(f"   - Test RMSE: {eval_results['rmse']:.4f}")
            logger.info(f"   - Model saved to: model/tgnn_plus_plus_best.pth")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the training test."""
    trainer = TGNNModelTrainingTest()
    success = trainer.run_complete_test()
    
    if success:
        print("\\n✅ Training test completed successfully!")
        print("📁 Check the 'results/' folder for plots and analysis")
        print("💾 Model saved in 'model/' folder")
    else:
        print("\\n❌ Training test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
