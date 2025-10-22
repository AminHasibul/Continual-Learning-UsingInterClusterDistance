"""
Example: Using ClusterAwareReplayFreeCL Strategy with RICS

This example demonstrates how to use the Cluster-Aware Replay-Free Continual
Learning strategy with RICS (Regularization via Inter-Cluster Separation) for
preventing catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models import SimpleCNN
from losses import InterClusterLoss
from experiments import ClusterAwareReplayFreeCL

# Check if Avalanche is installed
try:
    from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10
    from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
    from avalanche.logging import InteractiveLogger
    from avalanche.training.plugins import EvaluationPlugin
    AVALANCHE_AVAILABLE = True
except ImportError:
    AVALANCHE_AVAILABLE = False
    print("Warning: Avalanche not installed. Install with: pip install avalanche-lib")


def example_cluster_aware_cl_mnist():
    """
    Example using ClusterAwareReplayFreeCL on Split MNIST.
    
    This example:
    1. Creates a SimpleCNN model
    2. Sets up InterClusterLoss (RICS)
    3. Configures ClusterAwareReplayFreeCL strategy
    4. Trains on 5 MNIST tasks (2 classes each)
    5. Evaluates after each task
    """
    if not AVALANCHE_AVAILABLE:
        print("This example requires Avalanche. Please install it first.")
        return
    
    print("=" * 70)
    print("Cluster-Aware Replay-Free CL Example on Split MNIST")
    print("=" * 70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    model = SimpleCNN(in_channels=1, num_classes=10, feat_dim=256)
    print(f"Model: SimpleCNN with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create InterClusterLoss (RICS)
    inter_cluster_loss = InterClusterLoss(lambda_reg=1.0)
    print("InterClusterLoss (RICS) initialized")
    
    # Create evaluation plugin
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True),
        loggers=[interactive_logger]
    )
    
    # Create ClusterAwareReplayFreeCL strategy
    strategy = ClusterAwareReplayFreeCL(
        model=model,
        optimizer=optimizer,
        inter_cluster_loss=inter_cluster_loss,
        lambda_intra=1.0,      # Weight for RICS inter-cluster separation
        lambda_anchor=10.0,    # Weight for centroid anchoring
        lambda_logit=1.0,      # Weight for logit distillation
        lambda_var=1.0,        # Weight for variance anchoring
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=64,
        train_epochs=3,
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin
    )
    
    print("\nStrategy: ClusterAwareReplayFreeCL")
    print("  - lambda_intra (RICS): 1.0")
    print("  - lambda_anchor: 10.0")
    print("  - lambda_logit: 1.0")
    print("  - lambda_var: 1.0")
    
    # Create benchmark (Split MNIST: 5 tasks, 2 classes each)
    benchmark = SplitMNIST(n_experiences=5, seed=42)
    
    print(f"\nBenchmark: Split MNIST")
    print(f"  - Number of tasks: {len(benchmark.train_stream)}")
    print(f"  - Classes per task: 2")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        print(f"\n--- Task {task_id} ---")
        print(f"Classes in this task: {experience.classes_in_this_experience}")
        
        # Train on current task
        strategy.train(experience)
        
        # Evaluate on all seen tasks
        print(f"\nEvaluating on tasks 0-{task_id}...")
        strategy.eval(benchmark.test_stream[:task_id + 1])
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)


def example_cluster_aware_cl_cifar10():
    """
    Example using ClusterAwareReplayFreeCL on Split CIFAR-10.
    
    This example shows how to use the strategy on a more complex dataset.
    """
    if not AVALANCHE_AVAILABLE:
        print("This example requires Avalanche. Please install it first.")
        return
    
    print("=" * 70)
    print("Cluster-Aware Replay-Free CL Example on Split CIFAR-10")
    print("=" * 70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model (CIFAR-10 has 3 channels)
    model = SimpleCNN(in_channels=3, num_classes=10, feat_dim=256)
    print(f"Model: SimpleCNN with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create InterClusterLoss (RICS)
    inter_cluster_loss = InterClusterLoss(lambda_reg=1.0)
    
    # Create evaluation plugin
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True),
        loggers=[interactive_logger]
    )
    
    # Create strategy with adaptive regularization
    strategy = ClusterAwareReplayFreeCL(
        model=model,
        optimizer=optimizer,
        inter_cluster_loss=inter_cluster_loss,
        lambda_intra=1.0,      # RICS weight
        lambda_anchor=10.0,    # Decays with 0.5^task
        lambda_logit=1.0,      # Decays with 0.5^task
        lambda_var=1.0,        # Decays with 0.5^task
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=64,
        train_epochs=5,
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin
    )
    
    print("\nStrategy: ClusterAwareReplayFreeCL")
    print("  - Adaptive regularization: weights decay by 0.5^task")
    
    # Create benchmark (Split CIFAR-10: 5 tasks, 2 classes each)
    benchmark = SplitCIFAR10(n_experiences=5, seed=42)
    
    print(f"\nBenchmark: Split CIFAR-10")
    print(f"  - Number of tasks: {len(benchmark.train_stream)}")
    print(f"  - Classes per task: 2")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        print(f"\n--- Task {task_id} ---")
        print(f"Classes: {experience.classes_in_this_experience}")
        
        # Train
        strategy.train(experience)
        
        # Evaluate
        print(f"\nEvaluating on tasks 0-{task_id}...")
        strategy.eval(benchmark.test_stream[:task_id + 1])
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)


def example_basic_usage():
    """
    Basic usage example without Avalanche (for understanding the components).
    """
    print("=" * 70)
    print("Basic Usage Example (Understanding Components)")
    print("=" * 70)
    
    # 1. Create SimpleCNN model
    print("\n1. Creating SimpleCNN model...")
    model = SimpleCNN(in_channels=3, num_classes=10, feat_dim=256)
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    features, logits = model(x, return_feats=True)
    print(f"   Input: {x.shape}")
    print(f"   Features: {features.shape}")
    print(f"   Logits: {logits.shape}")
    
    # 2. Create InterClusterLoss
    print("\n2. Creating InterClusterLoss (RICS)...")
    icl = InterClusterLoss(lambda_reg=1.0)
    
    # Test with no centroids
    targets = torch.randint(0, 5, (4,))
    loss = icl(features, targets)
    print(f"   Loss (no centroids): {loss.item():.4f}")
    
    # Update centroids
    icl.update_centroids(features, targets)
    print(f"   Centroids updated for classes: {list(icl.class_centroids.keys())}")
    
    # Test with centroids
    features2 = torch.randn(4, 256)
    targets2 = torch.randint(5, 8, (4,))
    loss = icl(features2, targets2)
    print(f"   Loss (with centroids): {loss.item():.4f}")
    
    # 3. Explain the strategy
    print("\n3. ClusterAwareReplayFreeCL Strategy Components:")
    print("   a) Cross-Entropy Loss: Standard classification")
    print("   b) RICS (InterClusterLoss): Maximizes inter-cluster distance")
    print("   c) Centroid Anchoring: Keeps centroids stable across tasks")
    print("   d) Variance Anchoring: Preserves cluster variance")
    print("   e) Logit Distillation: Knowledge transfer from old model")
    print("   f) Feature Distillation: Representation preservation")
    
    print("\n4. Key Features:")
    print("   - Replay-free: No memory buffer needed")
    print("   - Adaptive decay: Regularization weights decrease over tasks")
    print("   - Multiple loss components: Comprehensive forgetting prevention")
    print("   - RICS integration: Uses normalized features for clustering")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("ClusterAwareReplayFreeCL with RICS Examples")
    print("="*70)
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "mnist":
            example_cluster_aware_cl_mnist()
        elif example == "cifar10":
            example_cluster_aware_cl_cifar10()
        elif example == "basic":
            example_basic_usage()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python example_cluster_aware_cl.py [mnist|cifar10|basic]")
    else:
        print("\nAvailable examples:")
        print("  1. python example_cluster_aware_cl.py basic")
        print("     - Basic component demonstration")
        print("\n  2. python example_cluster_aware_cl.py mnist")
        print("     - Full training on Split MNIST (requires Avalanche)")
        print("\n  3. python example_cluster_aware_cl.py cifar10")
        print("     - Full training on Split CIFAR-10 (requires Avalanche)")
        print("\nRunning basic example...\n")
        example_basic_usage()
