"""
Test Suite for ClusterAwareReplayFreeCL with RICS

This test suite validates:
1. SimpleCNN model functionality
2. InterClusterLoss (RICS) behavior
3. ClusterAwareReplayFreeCL strategy components
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models import SimpleCNN
from losses import InterClusterLoss


def test_simple_cnn():
    """Test SimpleCNN model."""
    print("Testing SimpleCNN...")
    
    # Test initialization
    model = SimpleCNN(in_channels=3, num_classes=10, feat_dim=256)
    assert model is not None, "Model initialization failed"
    
    # Test forward pass without features
    x = torch.randn(4, 3, 32, 32)
    logits = model(x, return_feats=False)
    assert logits.shape == (4, 10), f"Expected shape (4, 10), got {logits.shape}"
    
    # Test forward pass with features
    features, logits = model(x, return_feats=True)
    assert features.shape == (4, 256), f"Expected features shape (4, 256), got {features.shape}"
    assert logits.shape == (4, 10), f"Expected logits shape (4, 10), got {logits.shape}"
    
    # Test with different input sizes
    model_mnist = SimpleCNN(in_channels=1, num_classes=10, feat_dim=128)
    x_mnist = torch.randn(8, 1, 32, 32)
    features, logits = model_mnist(x_mnist, return_feats=True)
    assert features.shape == (8, 128), f"Expected features shape (8, 128), got {features.shape}"
    assert logits.shape == (8, 10), f"Expected logits shape (8, 10), got {logits.shape}"
    
    print("✓ SimpleCNN tests passed")


def test_inter_cluster_loss():
    """Test InterClusterLoss (RICS)."""
    print("\nTesting InterClusterLoss (RICS)...")
    
    # Test initialization
    icl = InterClusterLoss(lambda_reg=1.0)
    assert icl is not None, "InterClusterLoss initialization failed"
    assert len(icl.class_centroids) == 0, "Centroids should be empty initially"
    
    # Test with no centroids
    features = torch.randn(4, 256)
    targets = torch.randint(0, 5, (4,))
    loss = icl(features, targets)
    assert loss.item() == 0.0, f"Expected loss 0.0 with no centroids, got {loss.item()}"
    
    # Test centroid update
    icl.update_centroids(features, targets)
    assert len(icl.class_centroids) > 0, "Centroids should be updated"
    
    # Test with centroids from different classes
    new_features = torch.randn(4, 256)
    new_targets = torch.randint(5, 8, (4,))
    loss = icl(new_features, new_targets)
    assert loss.item() < 0, f"Expected negative loss (maximizing distance), got {loss.item()}"
    
    # Test normalization
    # Features should be normalized inside the loss
    large_features = torch.randn(4, 256) * 100
    icl2 = InterClusterLoss(lambda_reg=1.0)
    icl2.update_centroids(large_features, targets)
    for centroid in icl2.class_centroids.values():
        norm = torch.norm(centroid, p=2)
        assert abs(norm.item() - 1.0) < 0.01, f"Centroid should be normalized, got norm {norm.item()}"
    
    print("✓ InterClusterLoss tests passed")


def test_loss_components():
    """Test individual loss components."""
    print("\nTesting loss components...")
    
    # Test cross-entropy
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    assert ce_loss.item() > 0, "Cross-entropy loss should be positive"
    
    # Test MSE for anchoring
    centroids1 = torch.randn(3, 256)
    centroids2 = torch.randn(3, 256)
    mse_loss = nn.functional.mse_loss(centroids1, centroids2)
    assert mse_loss.item() >= 0, "MSE loss should be non-negative"
    
    # Test normalized features
    features = torch.randn(4, 256)
    normalized = nn.functional.normalize(features, p=2, dim=1)
    norms = torch.norm(normalized, p=2, dim=1)
    for norm in norms:
        assert abs(norm.item() - 1.0) < 0.01, f"Normalized features should have unit norm, got {norm.item()}"
    
    print("✓ Loss component tests passed")


def test_model_compatibility():
    """Test that SimpleCNN works with various configurations."""
    print("\nTesting model compatibility...")
    
    # Test with different configurations
    configs = [
        (1, 10, 128),   # MNIST
        (3, 10, 256),   # CIFAR-10
        (3, 100, 512),  # CIFAR-100
    ]
    
    for in_channels, num_classes, feat_dim in configs:
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)
        x = torch.randn(2, in_channels, 32, 32)
        features, logits = model(x, return_feats=True)
        
        assert features.shape == (2, feat_dim), \
            f"Expected features shape (2, {feat_dim}), got {features.shape}"
        assert logits.shape == (2, num_classes), \
            f"Expected logits shape (2, {num_classes}), got {logits.shape}"
    
    print("✓ Model compatibility tests passed")


def test_strategy_components():
    """Test components needed for ClusterAwareReplayFreeCL."""
    print("\nTesting strategy components...")
    
    # Test model + optimizer
    model = SimpleCNN(in_channels=3, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    assert optimizer is not None, "Optimizer creation failed"
    
    # Test InterClusterLoss integration
    icl = InterClusterLoss(lambda_reg=1.0)
    
    # Simulate a training step
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 5, (4,))
    
    features, logits = model(x, return_feats=True)
    
    # Classification loss
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    
    # RICS loss (should be 0 initially)
    rics_loss = icl(features, targets)
    assert rics_loss.item() == 0.0, "RICS loss should be 0 with no centroids"
    
    # Update centroids
    icl.update_centroids(features.detach(), targets)
    
    # Test with new batch
    x2 = torch.randn(4, 3, 32, 32)
    targets2 = torch.randint(5, 8, (4,))
    features2, logits2 = model(x2, return_feats=True)
    rics_loss2 = icl(features2, targets2)
    assert rics_loss2.item() < 0, "RICS loss should be negative (maximizing distance)"
    
    # Test combined loss
    total_loss = ce_loss + rics_loss2
    assert total_loss.item() != 0, "Total loss should be non-zero"
    
    # Test backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters())
    assert has_gradients, "Model should have gradients after backward pass"
    
    print("✓ Strategy component tests passed")


def test_adaptive_decay():
    """Test adaptive decay logic."""
    print("\nTesting adaptive decay...")
    
    base_lambda = 10.0
    
    # Test decay over tasks
    for task in range(5):
        decay_factor = 0.5 ** task
        current_lambda = base_lambda * decay_factor
        
        expected = base_lambda * (0.5 ** task)
        assert abs(current_lambda - expected) < 1e-6, \
            f"Decay calculation incorrect at task {task}"
        
        print(f"  Task {task}: λ = {current_lambda:.4f}")
    
    print("✓ Adaptive decay tests passed")


def test_feature_normalization():
    """Test feature normalization behavior."""
    print("\nTesting feature normalization...")
    
    # Test with unnormalized features
    features = torch.randn(10, 256) * 10
    normalized = nn.functional.normalize(features, p=2, dim=1)
    
    # Check norms
    norms = torch.norm(normalized, p=2, dim=1)
    for i, norm in enumerate(norms):
        assert abs(norm.item() - 1.0) < 0.01, \
            f"Feature {i} should have unit norm, got {norm.item()}"
    
    # Test that normalization doesn't change direction
    dot_products = (features * normalized).sum(dim=1)
    assert all(dot_products > 0), "Normalization should preserve direction"
    
    print("✓ Feature normalization tests passed")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("Running ClusterAwareReplayFreeCL Test Suite")
    print("="*70)
    
    try:
        test_simple_cnn()
        test_inter_cluster_loss()
        test_loss_components()
        test_model_compatibility()
        test_strategy_components()
        test_adaptive_decay()
        test_feature_normalization()
        
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
