"""
Cluster-Aware Replay-Free Continual Learning Strategy

This module implements a continual learning strategy that uses cluster-based
regularization with RICS (Regularization via Inter-Cluster Separation) to
prevent catastrophic forgetting without requiring a replay buffer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional

try:
    from avalanche.training.templates import SupervisedTemplate
except ImportError:
    raise ImportError(
        "This module requires Avalanche. Please install it with: pip install avalanche-lib"
    )


class ClusterAwareReplayFreeCL(SupervisedTemplate):
    """
    Cluster-Aware Replay-Free Continual Learning Strategy.
    
    This strategy prevents catastrophic forgetting by:
    1. Using RICS (Inter-Cluster Loss) to maximize separation between class clusters
    2. Anchoring current centroids to previous task centroids
    3. Preserving variance structure across tasks
    4. Using logit distillation from the previous model
    5. Using feature distillation from the previous model
    
    Args:
        model (nn.Module): The neural network model (must support return_feats=True)
        optimizer (torch.optim.Optimizer): Optimizer for training
        inter_cluster_loss (nn.Module): InterClusterLoss instance for RICS
        lambda_intra (float): Weight for intra-task cluster separation (default: 1.0)
        lambda_anchor (float): Weight for cross-task centroid anchoring (default: 10.0)
        lambda_logit (float): Weight for logit distillation (default: 1.0)
        lambda_var (float): Weight for variance anchoring (default: 1.0)
        criterion (nn.Module): Base classification loss (default: CrossEntropyLoss)
        **kwargs: Additional arguments for SupervisedTemplate
    """
    
    def __init__(
        self, 
        *, 
        model, 
        optimizer, 
        inter_cluster_loss,
        lambda_intra=1.0, 
        lambda_anchor=10.0, 
        lambda_logit=1.0, 
        lambda_var=1.0, 
        criterion=nn.CrossEntropyLoss(), 
        **kwargs
    ):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, **kwargs)
        
        # Save base values for decay
        self.base_lambda_anchor = lambda_anchor
        self.base_lambda_logit = lambda_logit
        self.base_lambda_var = lambda_var
        
        self.lambda_intra = lambda_intra
        self.lambda_anchor = lambda_anchor
        self.lambda_logit = lambda_logit
        self.lambda_var = lambda_var
        
        # RICS inter-cluster loss
        self.inter_cluster_loss = inter_cluster_loss
        
        # Storage for previous task information
        self.prev_centroids: Dict[int, torch.Tensor] = {}
        self.prev_variances: Dict[int, torch.Tensor] = {}
        self.model_old = None
        
        # Current batch information
        self.logits = None
        self.current_feats = None
    
    def forward(self):
        """Forward pass that extracts both features and logits."""
        self.current_feats, self.logits = self.model(self.mb_x, return_feats=True)
        return self.logits
    
    def criterion(self):
        """
        Compute the total loss combining multiple components:
        - Cross-entropy loss for classification
        - RICS inter-cluster loss for maximizing separation
        - Centroid anchoring loss for stability
        - Variance anchoring loss for consistency
        - Logit distillation loss for knowledge retention
        - Feature distillation loss for representation preservation
        """
        # Base classification loss
        ce_loss = self._criterion(self.logits, self.mb_y)
        
        # Normalize features for clustering
        feats = F.normalize(self.current_feats, p=2, dim=1)
        targets = self.mb_y
        
        # === Compute current centroids and variances ===
        curr_centroids, curr_variances = {}, {}
        for cls in torch.unique(targets):
            mask = (targets == cls)
            feats_cls = F.normalize(feats[mask], p=2, dim=1)
            curr_centroids[cls.item()] = feats_cls.mean(0)
            curr_variances[cls.item()] = feats_cls.var(0)
        
        # === RICS: Inter-cluster separation using InterClusterLoss ===
        # This replaces the manual inter_cluster_loss calculation
        rics_loss = self.inter_cluster_loss(self.current_feats, targets)
        
        # === Cross-task centroid anchoring ===
        anchor_loss = 0.0
        for cls, prev_mu in self.prev_centroids.items():
            if cls in curr_centroids:
                anchor_loss += F.mse_loss(curr_centroids[cls], prev_mu.to(feats.device))
        if self.prev_centroids:
            anchor_loss /= len(self.prev_centroids)
        
        # === Variance anchoring ===
        var_loss = 0.0
        for cls, prev_var in self.prev_variances.items():
            if cls in curr_variances:
                var_loss += F.mse_loss(curr_variances[cls], prev_var.to(feats.device))
        if self.prev_variances:
            var_loss /= len(self.prev_variances)
        
        # === Logit distillation ===
        logit_loss = 0.0
        if self.model_old is not None:
            with torch.no_grad():
                old_logits = self.model_old(self.mb_x)
            logit_loss = F.mse_loss(self.logits, old_logits)
        
        # === Feature distillation ===
        feat_loss = 0.0
        if self.model_old is not None:
            with torch.no_grad():
                old_feats, _ = self.model_old(self.mb_x, return_feats=True)
            feat_loss = F.mse_loss(self.current_feats, old_feats)
        
        # === Combine all losses ===
        return (
            ce_loss +
            self.lambda_intra * rics_loss +
            self.lambda_anchor * anchor_loss +
            self.lambda_logit * logit_loss +
            self.lambda_var * var_loss +
            0.001 * feat_loss  # small weight to avoid hurting plasticity
        )
    
    def _before_training_iteration(self, **kwargs):
        """Hook called before each training iteration."""
        self.logits = None
        self.current_feats = None
        current_task = self.experience.current_experience
        
        # Adaptive decay of regularization weights
        decay_factor = 0.5 ** current_task
        self.lambda_anchor = self.base_lambda_anchor * decay_factor
        self.lambda_logit = self.base_lambda_logit * decay_factor
        self.lambda_var = self.base_lambda_var * decay_factor
        
        super()._before_training_iteration(**kwargs)
    
    def _after_training_exp(self, **kwargs):
        """
        Hook called after training on each experience (task).
        Saves centroids, variances, and freezes the model.
        """
        print("[Cluster-Aware CL] Saving centroids, variances, and freezing model.")
        self.model.eval()
        dl = DataLoader(self.experience.dataset.eval(), batch_size=256, shuffle=False)
        centroids, variances = {}, {}
        
        with torch.no_grad():
            for x, y, *_ in dl:
                x, y = x.to(self.device), y.to(self.device)
                feats, _ = self.model(x, return_feats=True)
                feats = F.normalize(feats, p=2, dim=1)
                
                for cls in torch.unique(y):
                    mask = y == cls
                    feats_cls = feats[mask]
                    if mask.any():
                        if cls.item() not in centroids:
                            centroids[cls.item()] = []
                            variances[cls.item()] = []
                        centroids[cls.item()].append(feats_cls.mean(0))
                        variances[cls.item()].append(feats_cls.var(0))
        
        # Aggregate centroids and variances
        for cls in centroids:
            self.prev_centroids[cls] = torch.stack(centroids[cls]).mean(0).detach().cpu()
            self.prev_variances[cls] = torch.stack(variances[cls]).mean(0).detach().cpu()
        
        # Update InterClusterLoss centroids
        with torch.no_grad():
            all_feats = []
            all_targets = []
            for x, y, *_ in dl:
                x, y = x.to(self.device), y.to(self.device)
                feats, _ = self.model(x, return_feats=True)
                all_feats.append(feats)
                all_targets.append(y)
            
            if all_feats:
                all_feats = torch.cat(all_feats, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                self.inter_cluster_loss.update_centroids(all_feats, all_targets)
        
        # Freeze previous model for logit and feature distillation
        # Import the model class dynamically to avoid circular imports
        from models.simple_cnn import SimpleCNN
        
        self.model_old = SimpleCNN(
            in_channels=self.model.conv1.in_channels,
            num_classes=self.model.fc2.out_features,
            feat_dim=self.model.fc1.out_features
        )
        self.model_old.load_state_dict(self.model.state_dict())
        self.model_old.to(self.device)
        self.model_old.eval()
        
        print(f"[Freeze] Model frozen at task {self.experience.current_experience}")
        
        for p in self.model_old.parameters():
            p.requires_grad = False
        
        self.model.train()
        super()._after_training_exp(**kwargs)


if __name__ == "__main__":
    print("Cluster-Aware Replay-Free Continual Learning Strategy")
    print("=" * 60)
    print("\nThis module provides a continual learning strategy that uses")
    print("RICS (Regularization via Inter-Cluster Separation) to prevent")
    print("catastrophic forgetting without requiring a replay buffer.")
    print("\nExample usage:")
    print("""
    from models.simple_cnn import SimpleCNN
    from losses.custom_loss import InterClusterLoss
    from experiments.cluster_aware_cl import ClusterAwareReplayFreeCL
    import torch.optim as optim
    
    # Create model
    model = SimpleCNN(in_channels=3, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create InterClusterLoss for RICS
    inter_cluster_loss = InterClusterLoss(lambda_reg=1.0)
    
    # Create strategy
    strategy = ClusterAwareReplayFreeCL(
        model=model,
        optimizer=optimizer,
        inter_cluster_loss=inter_cluster_loss,
        lambda_intra=1.0,
        lambda_anchor=10.0,
        lambda_logit=1.0,
        lambda_var=1.0,
        train_mb_size=64,
        train_epochs=10,
        device='cuda'
    )
    
    # Train on continual learning benchmark
    # for experience in benchmark.train_stream:
    #     strategy.train(experience)
    #     strategy.eval(benchmark.test_stream)
    """)
