"""
XAI (Explainable AI) module implementing DAVOTS and ICFTS methods.
DAVOTS: Dynamic Attribution Visualization Over Time Series
ICFTS: Interventional Causal Framework for Time Series
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from captum.attr import IntegratedGradients, DeepLift, GradientShap
from captum.attr import LayerConductance, NoiseTunnel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DAVOTS:
    """
    Dynamic Attribution Visualization Over Time Series.
    Provides feature attribution analysis using gradient-based methods.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.deeplift = DeepLift(self.model)
        self.gradient_shap = GradientShap(self.model)
        
    def compute_attributions(self, input_tensor: torch.Tensor, 
                           baseline: Optional[torch.Tensor] = None,
                           method: str = 'integrated_gradients') -> torch.Tensor:
        """
        Compute feature attributions for input tensor.
        
        Args:
            input_tensor: Input data (batch_size, seq_len, features)
            baseline: Baseline for attribution (default: zeros)
            method: Attribution method ('integrated_gradients', 'deeplift', 'gradient_shap')
            
        Returns:
            Attribution scores with same shape as input
        """
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
            
        input_tensor = input_tensor.to(self.device)
        baseline = baseline.to(self.device)
        
        input_tensor.requires_grad_(True)
        
        if method == 'integrated_gradients':
            attributions = self.integrated_gradients.attribute(
                input_tensor, baseline, n_steps=50
            )
        elif method == 'deeplift':
            attributions = self.deeplift.attribute(input_tensor, baseline)
        elif method == 'gradient_shap':
            # Generate random baselines for GradientSHAP
            baselines = torch.randn(10, *input_tensor.shape[1:]).to(self.device)
            attributions = self.gradient_shap.attribute(
                input_tensor, baselines, n_samples=10
            )
        else:
            raise ValueError(f"Unknown attribution method: {method}")
            
        return attributions.detach()
    
    def compute_temporal_attributions(self, input_tensor: torch.Tensor, 
                                    feature_names: List[str],
                                    method: str = 'integrated_gradients') -> Dict[str, Any]:
        """
        Compute temporal attributions for visualization.
        
        Args:
            input_tensor: Input data (1, seq_len, features)
            feature_names: List of feature names
            method: Attribution method
            
        Returns:
            Dictionary with attribution results
        """
        
        # Compute attributions
        attributions = self.compute_attributions(input_tensor, method=method)
        
        # Convert to numpy for processing
        attr_np = attributions.cpu().numpy()[0]  # Remove batch dimension
        seq_len, num_features = attr_np.shape
        
        # Create attribution matrix (features × time)
        attr_matrix = attr_np.T  # Shape: (features, time)
        
        # Compute feature importance (sum over time)
        feature_importance = np.abs(attr_matrix).sum(axis=1)
        
        # Compute temporal importance (sum over features)
        temporal_importance = np.abs(attr_matrix).sum(axis=0)
        
        # Top contributing features
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]
        
        return {
            'attribution_matrix': attr_matrix,
            'feature_importance': feature_importance,
            'temporal_importance': temporal_importance,
            'top_features': top_features,
            'feature_names': feature_names,
            'method': method
        }
    
    def visualize_davots_heatmap(self, attribution_results: Dict[str, Any], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create DAVOTS heatmap visualization."""
        
        attr_matrix = attribution_results['attribution_matrix']
        feature_names = attribution_results['feature_names']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main heatmap
        sns.heatmap(attr_matrix, 
                   yticklabels=feature_names,
                   cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Attribution Score'},
                   ax=ax1)
        ax1.set_title('DAVOTS: Feature Attribution Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Features')
        
        # Feature importance bar plot
        top_features = attribution_results['top_features'][:15]  # Top 15
        feature_names_top = [f[0] for f in top_features]
        feature_scores = [f[1] for f in top_features]
        
        ax2.barh(range(len(feature_names_top)), feature_scores)
        ax2.set_yticks(range(len(feature_names_top)))
        ax2.set_yticklabels(feature_names_top)
        ax2.set_xlabel('Total Attribution Score')
        ax2.set_title('Top Contributing Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"DAVOTS heatmap saved to {save_path}")
            
        return fig

class ICFTS:
    """
    Interventional Causal Framework for Time Series.
    Provides causal analysis using interventional methods and counterfactuals.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def compute_causal_effects(self, input_tensor: torch.Tensor, 
                             feature_names: List[str],
                             intervention_strength: float = 0.1) -> Dict[str, Any]:
        """
        Compute causal effects using interventional analysis.
        
        Args:
            input_tensor: Input data (1, seq_len, features)
            feature_names: List of feature names
            intervention_strength: Strength of intervention (as fraction of std)
            
        Returns:
            Dictionary with causal analysis results
        """
        
        input_tensor = input_tensor.to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_pred = self.model(input_tensor).cpu().numpy()[0, 0]
        
        causal_effects = {}
        feature_effects = []
        
        for i, feature_name in enumerate(feature_names):
            # Create intervention: modify feature i
            intervened_tensor = input_tensor.clone()
            
            # Compute intervention magnitude
            feature_std = input_tensor[0, :, i].std()
            intervention_magnitude = intervention_strength * feature_std
            
            # Apply positive intervention
            intervened_tensor[0, :, i] += intervention_magnitude
            
            with torch.no_grad():
                pos_pred = self.model(intervened_tensor).cpu().numpy()[0, 0]
            
            # Apply negative intervention
            intervened_tensor = input_tensor.clone()
            intervened_tensor[0, :, i] -= intervention_magnitude
            
            with torch.no_grad():
                neg_pred = self.model(intervened_tensor).cpu().numpy()[0, 0]
            
            # Compute causal effect
            causal_effect = (pos_pred - neg_pred) / (2 * intervention_magnitude.item())
            
            causal_effects[feature_name] = {
                'effect': causal_effect,
                'pos_pred': pos_pred,
                'neg_pred': neg_pred,
                'baseline': baseline_pred
            }
            
            feature_effects.append((feature_name, abs(causal_effect)))
        
        # Sort by effect magnitude
        feature_effects.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'causal_effects': causal_effects,
            'ranked_effects': feature_effects,
            'baseline_prediction': baseline_pred,
            'intervention_strength': intervention_strength
        }
    
    def compute_counterfactuals(self, input_tensor: torch.Tensor,
                              target_change: float,
                              feature_names: List[str],
                              max_iterations: int = 100) -> Dict[str, Any]:
        """
        Compute counterfactual explanations.
        
        Args:
            input_tensor: Input data (1, seq_len, features)
            target_change: Desired change in prediction
            feature_names: List of feature names
            max_iterations: Maximum optimization iterations
            
        Returns:
            Counterfactual results
        """
        
        input_tensor = input_tensor.to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_pred = self.model(input_tensor).cpu().numpy()[0, 0]
        
        target_pred = baseline_pred + target_change
        
        # Initialize counterfactual
        counterfactual = input_tensor.clone().detach()
        counterfactual.requires_grad_(True)
        
        optimizer = torch.optim.Adam([counterfactual], lr=0.01)
        
        best_loss = float('inf')
        best_counterfactual = None
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Prediction loss
            pred = self.model(counterfactual)
            pred_loss = (pred[0, 0] - target_pred) ** 2
            
            # Proximity loss (L2 distance from original)
            proximity_loss = torch.sum((counterfactual - input_tensor) ** 2)
            
            # Total loss
            total_loss = pred_loss + 0.1 * proximity_loss
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_counterfactual = counterfactual.clone().detach()
            
            if iteration % 20 == 0:
                current_pred = pred.item()
                logger.debug(f"Iteration {iteration}: Pred={current_pred:.4f}, Target={target_pred:.4f}")
        
        # Analyze changes
        if best_counterfactual is not None:
            changes = (best_counterfactual - input_tensor).cpu().numpy()[0]  # (seq_len, features)
            
            # Compute feature change magnitudes
            feature_changes = np.abs(changes).sum(axis=0)  # Sum over time
            
            # Top changed features
            top_changes_idx = np.argsort(feature_changes)[-10:][::-1]
            top_changes = [(feature_names[i], feature_changes[i]) for i in top_changes_idx]
            
            # Final prediction
            with torch.no_grad():
                final_pred = self.model(best_counterfactual).cpu().numpy()[0, 0]
        else:
            changes = None
            feature_changes = None
            top_changes = []
            final_pred = baseline_pred
        
        return {
            'counterfactual': best_counterfactual,
            'changes': changes,
            'feature_changes': feature_changes,
            'top_changes': top_changes,
            'baseline_prediction': baseline_pred,
            'final_prediction': final_pred,
            'target_prediction': target_pred,
            'success': abs(final_pred - target_pred) < abs(target_change) * 0.1
        }
    
    def visualize_causal_effects(self, causal_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Visualize causal effects."""
        
        ranked_effects = causal_results['ranked_effects'][:15]  # Top 15
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Causal effects bar plot
        feature_names = [f[0] for f in ranked_effects]
        effect_magnitudes = [f[1] for f in ranked_effects]
        
        ax1.barh(range(len(feature_names)), effect_magnitudes)
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('Causal Effect Magnitude')
        ax1.set_title('ICFTS: Top Causal Effects')
        
        # Intervention results scatter plot
        causal_effects = causal_results['causal_effects']
        baseline = causal_results['baseline_prediction']
        
        pos_preds = [causal_effects[f]['pos_pred'] - baseline for f in feature_names]
        neg_preds = [causal_effects[f]['neg_pred'] - baseline for f in feature_names]
        
        ax2.scatter(pos_preds, neg_preds, alpha=0.6)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Positive Intervention Effect')
        ax2.set_ylabel('Negative Intervention Effect')
        ax2.set_title('Intervention Effects Scatter')
        
        # Add diagonal line
        lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
                max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
        ax2.plot(lims, [-x for x in lims], 'r--', alpha=0.5, label='y = -x')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ICFTS visualization saved to {save_path}")
            
        return fig

class XAIAnalyzer:
    """Combined XAI analyzer using both DAVOTS and ICFTS."""
    
    def __init__(self, model, feature_names: List[str], 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        
        self.davots = DAVOTS(model, device)
        self.icfts = ICFTS(model, device)
        
    def full_analysis(self, input_tensor: torch.Tensor,
                     save_dir: str = "data/xai",
                     target_change: float = 0.05) -> Dict[str, Any]:
        """
        Perform complete XAI analysis using both DAVOTS and ICFTS.
        
        Args:
            input_tensor: Input data (1, seq_len, features)
            save_dir: Directory to save visualizations
            target_change: Target change for counterfactual analysis
            
        Returns:
            Complete analysis results
        """
        
        logger.info("Starting comprehensive XAI analysis...")
        
        # DAVOTS analysis
        logger.info("Computing DAVOTS attributions...")
        davots_results = self.davots.compute_temporal_attributions(
            input_tensor, self.feature_names, method='integrated_gradients'
        )
        
        # ICFTS analysis
        logger.info("Computing ICFTS causal effects...")
        icfts_results = self.icfts.compute_causal_effects(
            input_tensor, self.feature_names
        )
        
        # Counterfactual analysis
        logger.info("Computing counterfactuals...")
        counterfactual_results = self.icfts.compute_counterfactuals(
            input_tensor, target_change, self.feature_names
        )
        
        # Create visualizations
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        davots_fig = self.davots.visualize_davots_heatmap(
            davots_results, save_path=os.path.join(save_dir, "davots_heatmap.png")
        )
        
        icfts_fig = self.icfts.visualize_causal_effects(
            icfts_results, save_path=os.path.join(save_dir, "icfts_effects.png")
        )
        
        # Save numerical results
        np.save(os.path.join(save_dir, "davots_matrix.npy"), 
                davots_results['attribution_matrix'])
        
        np.save(os.path.join(save_dir, "icfts_effects.npy"),
                np.array([f[1] for f in icfts_results['ranked_effects']]))
        
        # Combined results
        results = {
            'davots': davots_results,
            'icfts': icfts_results,
            'counterfactuals': counterfactual_results,
            'visualizations': {
                'davots_fig': davots_fig,
                'icfts_fig': icfts_fig
            }
        }
        
        logger.info("XAI analysis complete!")
        return results
    
    def generate_explanation_text(self, analysis_results: Dict[str, Any]) -> str:
        """Generate human-readable explanation text."""
        
        davots = analysis_results['davots']
        icfts = analysis_results['icfts']
        
        explanation = []
        explanation.append("=== Stock Prediction Explanation ===\n")
        
        # Top contributing features
        explanation.append("TOP CONTRIBUTING FEATURES (DAVOTS):")
        for i, (feature, score) in enumerate(davots['top_features'][:5]):
            explanation.append(f"{i+1}. {feature}: {score:.4f}")
        
        explanation.append("\nSTRONGEST CAUSAL EFFECTS (ICFTS):")
        for i, (feature, effect) in enumerate(icfts['ranked_effects'][:5]):
            explanation.append(f"{i+1}. {feature}: {effect:.4f}")
        
        explanation.append(f"\nBaseline Prediction: {icfts['baseline_prediction']:.4f}")
        
        return "\n".join(explanation)

# Example usage and testing
def test_xai_analysis():
    """Test XAI analysis functionality."""
    
    # Create dummy model for testing
    from model.tgnn_plus_plus import TGNNPlusPlus
    
    input_dim = 50
    model = TGNNPlusPlus(input_dim, hidden_dim=64, num_layers=1)
    
    # Create dummy data
    test_input = torch.randn(1, 30, input_dim)
    feature_names = [f"feature_{i}" for i in range(input_dim)]
    
    # Initialize XAI analyzer
    xai_analyzer = XAIAnalyzer(model, feature_names, device='cpu')
    
    # Run analysis
    results = xai_analyzer.full_analysis(test_input, save_dir="test_xai")
    
    # Generate explanation
    explanation = xai_analyzer.generate_explanation_text(results)
    print(explanation)
    
    return results

if __name__ == "__main__":
    test_xai_analysis()
