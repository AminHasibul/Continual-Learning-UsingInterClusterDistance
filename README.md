# Continual Learning: Preventing Catastrophic Forgetting

A comprehensive framework for continual learning research with custom neural network architectures and loss functions designed to prevent catastrophic forgetting.

## Overview

This repository provides a modular and extensible framework for experimenting with continual learning (also known as lifelong learning or incremental learning). The goal is to enable neural networks to learn from sequential tasks while retaining knowledge from previous tasks.

### Key Features

- **Custom Architectures**: Implementations of CNN and ResNet architectures optimized for continual learning
- **Advanced Loss Functions**: Multiple loss functions including Knowledge Distillation, EWC, LwF, and iCaRL
- **Benchmark Framework**: Tools for evaluating continual learning methods with comprehensive metrics
- **Jupyter Notebook**: Interactive starter notebook with experiments and visualizations

## Project Structure

```
Continual-Learning-/
├── models/
│   ├── __init__.py
│   ├── custom_cnn.py              # Custom CNN architecture
│   └── custom_resnet.py           # Custom ResNet architecture with continual learning features
├── losses/
│   ├── __init__.py
│   └── custom_loss.py             # Custom loss functions (KD, EWC, LwF, iCaRL)
├── experiments/
│   ├── __init__.py
│   └── continual_learning_benchmark.py  # Benchmark framework and utilities
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

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AminHasibul/Continual-Learning-.git
cd Continual-Learning-
```

2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib seaborn jupyter
```

## Quick Start

### Using Python Scripts

```python
from models import CustomCNN, create_resnet18
from losses import KnowledgeDistillationLoss, ElasticWeightConsolidationLoss
from experiments import ContinualLearningBenchmark

# Create a model
model = CustomCNN(num_classes=10, input_channels=3)

# Create benchmark
benchmark = ContinualLearningBenchmark(
    model=model,
    device='cuda',
    num_tasks=5
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

## Continual Learning Methods

This repository implements several state-of-the-art continual learning techniques:

1. **Knowledge Distillation**: Preserve old knowledge by distilling it during new task learning
2. **Elastic Weight Consolidation (EWC)**: Protect important weights using Fisher Information
3. **Learning without Forgetting (LwF)**: Use knowledge distillation to maintain old task performance
4. **iCaRL**: Incremental classifier using nearest-mean-of-exemplars and distillation

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
