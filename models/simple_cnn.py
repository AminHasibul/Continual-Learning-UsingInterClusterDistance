"""
Simple CNN Architecture for Continual Learning with Feature Extraction

This module implements a simple Convolutional Neural Network designed for
continual learning scenarios with support for feature extraction which is
required for RICS (Regularization via Inter-Cluster Separation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for continual learning with feature extraction.
    
    This is a two-layer CNN with pooling, then a 256-unit dense layer. 
    The return_feats parameter lets us extract the feature vector for 
    RICS clustering and other continual learning techniques.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        num_classes (int): Number of output classes (default: 10)
        feat_dim (int): Dimension of feature layer (default: 256)
    """
    
    def __init__(self, in_channels=3, num_classes=10, feat_dim=256):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, feat_dim)
        self.fc2 = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x, return_feats=False):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            return_feats (bool): If True, returns both features and logits
            
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
                If return_feats=False: Output logits of shape (batch_size, num_classes)
                If return_feats=True: Tuple of (features, logits)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        feats = F.relu(self.fc1(x))
        logits = self.fc2(feats)
        return (feats, logits) if return_feats else logits


if __name__ == "__main__":
    # Test the model
    print("Testing SimpleCNN:")
    model = SimpleCNN(in_channels=3, num_classes=10)
    print(model)
    
    # Test with random input
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4, 32x32 RGB images
    
    # Test normal forward
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with feature extraction
    features, logits = model(dummy_input, return_feats=True)
    print(f"Feature shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")
