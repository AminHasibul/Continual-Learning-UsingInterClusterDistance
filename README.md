# Continual Learning: Preventing Catastrophic Forgetting

A comprehensive framework for continual learning research with custom neural network architectures and loss functions designed to prevent catastrophic forgetting.

## Overview

This repository provides a modular and extensible framework for experimenting with continual learning (also known as lifelong learning or incremental learning). The goal is to enable neural networks to learn from sequential tasks while retaining knowledge from previous tasks.

### Key Features

- **Custom Architectures**: Implementations of CNN and ResNet architectures optimized for continual learning
  - SimpleCNN with feature extraction support for RICS
- **Advanced Loss Functions**: Multiple loss functions including Knowledge Distillation, EWC, LwF, iCaRL, and RICS
  - InterClusterLoss (RICS) for maximizing inter-cluster separation
- **Continual Learning Strategies**: Ready-to-use strategies for preventing catastrophic forgetting
  - ClusterAwareReplayFreeCL with RICS integration
- **Benchmark Framework**: Tools for evaluating continual learning methods with comprehensive metrics
- **Jupyter Notebook**: Interactive starter notebook with experiments and visualizations

## Project Structure

```
Continual-Learning-/
├── models/
│   ├── __init__.py
│   ├── custom_cnn.py              # Custom CNN architecture
│   ├── simple_cnn.py               # SimpleCNN with feature extraction for RICS
│   └── custom_resnet.py           # Custom ResNet architecture with continual learning features
├── losses/
│   ├── __init__.py
│   └── custom_loss.py             # Custom loss functions (KD, EWC, LwF, iCaRL, RICS)
├── experiments/
│   ├── __init__.py
│   ├── continual_learning_benchmark.py  # Benchmark framework and utilities
│   └── cluster_aware_cl.py        # ClusterAwareReplayFreeCL strategy with RICS
├── example_cluster_aware_cl.py    # Example usage of ClusterAwareReplayFreeCL
├── Preventing_Forgetting_Continual_Learning.ipynb  # Starter notebook
├── Solving_Catastrophic_fogetting_for_continual_learning_using_custom_ResNET_and_ICF.ipynb
├── README.md
└── LICENSE
```

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- seaborn
- jupyter
- avalanche-lib (for continual learning strategies)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AminHasibul/Continual-Learning-.git
cd Continual-Learning-
```

2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib seaborn jupyter avalanche-lib
```

## Quick Start

### Using Python Scripts

```python
from models import CustomCNN, SimpleCNN, create_resnet18
from losses import KnowledgeDistillationLoss, ElasticWeightConsolidationLoss, InterClusterLoss
from experiments import ContinualLearningBenchmark, ClusterAwareReplayFreeCL

# Create a model
model = CustomCNN(num_classes=10, input_channels=3)

# Create benchmark
benchmark = ContinualLearningBenchmark(
    model=model,
    device='cuda',
    num_tasks=5
)

# Or use ClusterAwareReplayFreeCL with RICS
model = SimpleCNN(in_channels=3, num_classes=10)
inter_cluster_loss = InterClusterLoss(lambda_reg=1.0)
strategy = ClusterAwareReplayFreeCL(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    inter_cluster_loss=inter_cluster_loss,
    lambda_intra=1.0,
    lambda_anchor=10.0,
    lambda_logit=1.0,
    lambda_var=1.0,
    device='cuda'
)

# Run experiments (see notebook for detailed examples)
```

### Using the Jupyter Notebook

Launch the starter notebook to explore interactive examples:

```bash
jupyter notebook Preventing_Forgetting_Continual_Learning.ipynb
```

The notebook includes:
- Complete setup and data loading
- Baseline experiments
- Multiple continual learning methods (KD, EWC, LwF)
- Visualizations and performance comparisons
- Metrics analysis (accuracy, forgetting, backward transfer)

## Components

### Models (`models/`)

#### CustomCNN
A flexible convolutional neural network with:
- Multiple convolutional blocks with batch normalization
- Dropout for regularization
- Feature extraction capabilities
- Customizable architecture

#### SimpleCNN
A simple CNN designed for continual learning with:
- Two convolutional layers with pooling
- 256-dimensional feature layer
- Support for feature extraction (return_feats=True)
- Optimized for use with RICS and clustering methods

#### CustomResNet
ResNet-style architecture with:
- Residual blocks with skip connections
- Support for ResNet-18 and ResNet-34 configurations
- Layer-wise parameter access for importance-based methods
- Feature extraction for continual learning

### Loss Functions (`losses/`)

#### KnowledgeDistillationLoss
Transfers knowledge from a teacher model to a student model while learning new tasks.

#### ElasticWeightConsolidationLoss (EWC)
Constrains important parameters to stay close to their values from previous tasks using Fisher Information Matrix.

#### LwFLoss (Learning without Forgetting)
Preserves performance on old tasks using knowledge distillation during new task learning.

#### iCaRLLoss
Implements the iCaRL approach with sigmoid cross-entropy for incremental learning.

#### InterClusterLoss (RICS)
Regularization via Inter-Cluster Separation that:
- Maximizes Euclidean distance between normalized feature clusters
- Pushes new class features away from old class centroids
- Prevents catastrophic forgetting through cluster separation
- Uses normalized features to avoid loss explosion

### Experiments (`experiments/`)

#### ContinualLearningBenchmark
A comprehensive framework for:
- Sequential task training
- Multi-task evaluation
- Performance tracking over time
- Computing continual learning metrics:
  - Average accuracy
  - Forgetting measure
  - Backward transfer

#### ClusterAwareReplayFreeCL
A replay-free continual learning strategy that prevents catastrophic forgetting using:
- **RICS (InterClusterLoss)**: Maximizes inter-cluster distance between classes
- **Centroid Anchoring**: Keeps class centroids stable across tasks
- **Variance Anchoring**: Preserves cluster variance structure
- **Logit Distillation**: Transfers knowledge from previous model
- **Feature Distillation**: Preserves learned representations
- **Adaptive Decay**: Regularization weights decrease over tasks (0.5^task)

Key advantages:
- No replay buffer required (memory-efficient)
- Multiple complementary regularization techniques
- Works with SimpleCNN and models supporting return_feats=True

## Continual Learning Methods

This repository implements several state-of-the-art continual learning techniques:

1. **Knowledge Distillation**: Preserve old knowledge by distilling it during new task learning
2. **Elastic Weight Consolidation (EWC)**: Protect important weights using Fisher Information
3. **Learning without Forgetting (LwF)**: Use knowledge distillation to maintain old task performance
4. **iCaRL**: Incremental classifier using nearest-mean-of-exemplars and distillation
5. **RICS (Cluster-Aware Replay-Free)**: Maximize inter-cluster separation with multiple regularization techniques
   - Inter-cluster loss for class separation
   - Centroid and variance anchoring
   - Logit and feature distillation
   - Adaptive regularization decay

## Quick Start Examples

### Using ClusterAwareReplayFreeCL with RICS

See `example_cluster_aware_cl.py` for complete examples:

```bash
# Basic component demonstration
python example_cluster_aware_cl.py basic

# Full training on Split MNIST (requires Avalanche)
python example_cluster_aware_cl.py mnist

# Full training on Split CIFAR-10 (requires Avalanche)
python example_cluster_aware_cl.py cifar10
```

### Basic Usage

```python
from models import SimpleCNN
from losses import InterClusterLoss
from experiments import ClusterAwareReplayFreeCL
import torch.optim as optim

# Create model with feature extraction support
model = SimpleCNN(in_channels=3, num_classes=10, feat_dim=256)

# Create InterClusterLoss (RICS)
inter_cluster_loss = InterClusterLoss(lambda_reg=1.0)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create strategy
strategy = ClusterAwareReplayFreeCL(
    model=model,
    optimizer=optimizer,
    inter_cluster_loss=inter_cluster_loss,
    lambda_intra=1.0,      # Weight for RICS
    lambda_anchor=10.0,    # Weight for centroid anchoring
    lambda_logit=1.0,      # Weight for logit distillation
    lambda_var=1.0,        # Weight for variance anchoring
    train_mb_size=64,
    train_epochs=10,
    device='cuda'
)

# Train on continual learning benchmark (requires Avalanche)
# for experience in benchmark.train_stream:
#     strategy.train(experience)
#     strategy.eval(benchmark.test_stream)
```

## Metrics

The framework computes standard continual learning metrics:

- **Average Accuracy**: Mean accuracy across all tasks
- **Forgetting**: Measures how much performance degrades on old tasks
- **Backward Transfer**: Measures if learning new tasks helps old task performance


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{continual-learning-framework,
  author = {Amin Hasibul},
  title = {Continual Learning: Preventing Catastrophic Forgetting},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/AminHasibul/Continual-Learning-}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
