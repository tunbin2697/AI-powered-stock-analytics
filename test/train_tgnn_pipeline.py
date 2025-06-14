"""
Comprehensive training script for TGNN++ with XAI integration.
Combines data integration, model training, and XAI analysis.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTrainer:
    """Comprehensive trainer for TGNN++ with XAI capabilities."""
    
    def __init__(self, config: dict = None):
        self.config = config or self.get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.data_integrator = None
        self.model = None
        self.trainer = None
        self.xai_analyzer = None
        
        # Create directories
        os.makedirs('model', exist_ok=True)
        os.makedirs('data/xai', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'symbol': 'AAPL',
            'date_suffix': '2025-06-14',
            'sequence_length': 30,
            'batch_size': 32,
            'train_split': 0.8,
            'hidden_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 20,
            'target_column': 'Close'
        }
    
    def step1_integrate_data(self) -> dict:
        """Step 1: Integrate multi-modal data."""
        logger.info("=== Step 1: Data Integration ===")
        
        self.data_integrator = MultiModalDataIntegrator()
        
        # Load processed data
        data_info = self.data_integrator.load_processed_data(
            symbol=self.config['symbol'],
            date_suffix=self.config['date_suffix']
        )
        logger.info(f"Data loading info: {data_info}")
        
        # Align data by date
        if data_info['price_shape'] is not None:
            aligned_data = self.data_integrator.align_data_by_date()
            logger.info(f"Aligned data shape: {aligned_data.shape}")
            
            # Save integrated data
            self.data_integrator.save_integrated_data("data/processed/integrated_multimodal_data.csv")
            
            return {
                'status': 'success',
                'data_shape': aligned_data.shape,
                'num_features': len(self.data_integrator.get_feature_names()) + 1,  # +1 for target
                'feature_names': self.data_integrator.get_feature_names()
            }
        else:
            return {'status': 'error', 'message': 'No price data available'}
    
    def step2_prepare_sequences(self, integration_result: dict) -> dict:
        """Step 2: Prepare sequences for training."""
        logger.info("=== Step 2: Sequence Preparation ===")
        
        if integration_result['status'] != 'success':
            return {'status': 'error', 'message': 'Data integration failed'}
        
        # Create sequences
        X_train, X_test, y_train, y_test = self.data_integrator.create_sequences(
            sequence_length=self.config['sequence_length'],
            target_col=self.config['target_column'],
            train_split=self.config['train_split']
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Store for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        return {
            'status': 'success',
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'input_dim': X_train.shape[2],
            'sequence_length': X_train.shape[1]
        }
    
    def step3_build_model(self, sequence_result: dict) -> dict:
        """Step 3: Build TGNN++ model."""
        logger.info("=== Step 3: Model Building ===")
        
        if sequence_result['status'] != 'success':
            return {'status': 'error', 'message': 'Sequence preparation failed'}
        
        input_dim = sequence_result['input_dim']
        
        # Create model
        self.model = TGNNPlusPlus(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            output_dim=1
        )
        
        # Create trainer
        self.trainer = TGNNTrainer(self.model, device=self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return {
            'status': 'success',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_dim': input_dim
        }
    
    def step4_train_model(self, model_result: dict) -> dict:
        """Step 4: Train the model."""
        logger.info("=== Step 4: Model Training ===")
        
        if model_result['status'] != 'success':
            return {'status': 'error', 'message': 'Model building failed'}
        
        # Train the model
        train_losses, val_losses = self.trainer.train(
            self.train_loader,
            self.test_loader,
            epochs=self.config['epochs'],
            early_stopping_patience=self.config['early_stopping_patience']
        )
        
        # Evaluate on test set
        test_loss = self.trainer.validate(self.test_loader)
        
        # Generate predictions for evaluation
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())
        
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Save training curves
        self.plot_training_curves(train_losses, val_losses)
        
        # Save predictions vs actual
        self.plot_predictions(targets, predictions)
        
        return {
            'status': 'success',
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'num_epochs': len(train_losses)
        }
    
    def step5_xai_analysis(self, training_result: dict) -> dict:
        """Step 5: Perform XAI analysis."""
        logger.info("=== Step 5: XAI Analysis ===")
        
        if training_result['status'] != 'success':
            return {'status': 'error', 'message': 'Model training failed'}
        
        # Initialize XAI analyzer
        feature_names = self.data_integrator.get_feature_names()
        self.xai_analyzer = XAIAnalyzer(self.model, feature_names, device=self.device)
        
        # Get a sample for XAI analysis (latest window)
        sample_input = self.data_integrator.get_latest_window(
            window_size=self.config['sequence_length']
        )
        
        # Perform comprehensive XAI analysis
        xai_results = self.xai_analyzer.full_analysis(
            sample_input,
            save_dir="data/xai",
            target_change=0.05
        )
        
        # Generate explanation text
        explanation_text = self.xai_analyzer.generate_explanation_text(xai_results)
        
        # Save explanation
        with open("data/xai/explanation.txt", "w") as f:
            f.write(explanation_text)
        
        logger.info("XAI analysis completed")
        logger.info("Explanation preview:")
        logger.info(explanation_text[:500] + "...")
        
        return {
            'status': 'success',
            'top_davots_features': xai_results['davots']['top_features'][:5],
            'top_causal_features': xai_results['icfts']['ranked_effects'][:5],
            'baseline_prediction': xai_results['icfts']['baseline_prediction'],
            'explanation_path': "data/xai/explanation.txt"
        }
    
    def step6_save_results(self, xai_result: dict) -> dict:
        """Step 6: Save all results and create summary."""
        logger.info("=== Step 6: Results Summary ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive results summary
        summary = {
            'timestamp': timestamp,
            'config': self.config,
            'data_integration': {
                'status': 'success',
                'final_shape': self.data_integrator.combined_data.shape,
                'num_features': len(self.data_integrator.get_feature_names())
            },
            'model_performance': {
                'test_mse': float(self.test_mse) if hasattr(self, 'test_mse') else None,
                'test_mae': float(self.test_mae) if hasattr(self, 'test_mae') else None,
                'test_r2': float(self.test_r2) if hasattr(self, 'test_r2') else None
            },
            'xai_analysis': xai_result if xai_result['status'] == 'success' else None
        }
        
        # Save summary
        import json
        with open(f"results/training_summary_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_names': self.data_integrator.get_feature_names(),
            'timestamp': timestamp
        }, f"model/tgnn_plus_plus_final_{timestamp}.pth")
        
        logger.info(f"Results saved with timestamp: {timestamp}")
        
        return {
            'status': 'success',
            'timestamp': timestamp,
            'summary_path': f"results/training_summary_{timestamp}.json",
            'model_path': f"model/tgnn_plus_plus_final_{timestamp}.pth"
        }
    
    def plot_training_curves(self, train_losses: list, val_losses: list):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_predictions(self, targets: np.ndarray, predictions: np.ndarray):
        """Plot predictions vs actual values."""
        plt.figure(figsize=(12, 5))
        
        # Time series plot
        plt.subplot(1, 2, 1)
        plt.plot(targets, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Prediction Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete training pipeline."""
        logger.info("Starting complete TGNN++ training pipeline...")
        
        results = {}
        
        # Step 1: Data Integration
        results['step1'] = self.step1_integrate_data()
        if results['step1']['status'] != 'success':
            return results
        
        # Step 2: Sequence Preparation
        results['step2'] = self.step2_prepare_sequences(results['step1'])
        if results['step2']['status'] != 'success':
            return results
        
        # Step 3: Model Building
        results['step3'] = self.step3_build_model(results['step2'])
        if results['step3']['status'] != 'success':
            return results
        
        # Step 4: Model Training
        results['step4'] = self.step4_train_model(results['step3'])
        if results['step4']['status'] != 'success':
            return results
        
        # Store metrics for summary
        self.test_mse = results['step4']['mse']
        self.test_mae = results['step4']['mae']
        self.test_r2 = results['step4']['r2']
        
        # Step 5: XAI Analysis
        results['step5'] = self.step5_xai_analysis(results['step4'])
        if results['step5']['status'] != 'success':
            logger.warning("XAI analysis failed, but continuing...")
        
        # Step 6: Save Results
        results['step6'] = self.step6_save_results(results['step5'])
        
        logger.info("=== Pipeline Complete ===")
        logger.info(f"Final Model Performance:")
        logger.info(f"  MSE: {self.test_mse:.6f}")
        logger.info(f"  MAE: {self.test_mae:.6f}")
        logger.info(f"  R²: {self.test_r2:.6f}")
        
        return results

def main():
    """Main training function."""
    # Custom configuration if needed
    config = {
        'symbol': 'AAPL',
        'date_suffix': '2025-06-14',
        'sequence_length': 30,
        'batch_size': 16,  # Reduced for stability
        'train_split': 0.8,
        'hidden_dim': 64,  # Reduced for faster training
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 50,  # Reduced for testing
        'early_stopping_patience': 15,
        'target_column': 'Close'
    }
    
    # Create trainer
    trainer = ComprehensiveTrainer(config)
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline()
    
    # Print final summary
    print("\n" + "="*50)
    print("TRAINING PIPELINE SUMMARY")
    print("="*50)
    
    for step, result in results.items():
        print(f"{step.upper()}: {result['status']}")
        if result['status'] == 'error':
            print(f"  Error: {result.get('message', 'Unknown error')}")
            break
    
    print("="*50)
    
    return results

if __name__ == "__main__":
    main()
