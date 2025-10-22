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

## Example Results

The performance matrix shows accuracy on each task (columns) after training on task i (rows):

```
Performance Matrix:
      T 0  T 1  T 2  T 3  T 4
T 0 | 85.0    -    -    -    -
T 1 | 82.0 87.0    -    -    -
T 2 | 78.0 83.0 86.0    -    -
T 3 | 75.0 80.0 82.0 88.0    -
T 4 | 72.0 77.0 79.0 84.0 89.0
```

## Customization

### Adding New Architectures

Create a new file in `models/` and inherit from `nn.Module`:

```python
import torch.nn as nn

class YourCustomModel(nn.Module):
    def __init__(self, num_classes):
        super(YourCustomModel, self).__init__()
        # Define your architecture
        
    def forward(self, x):
        # Define forward pass
        return x
```

### Adding New Loss Functions

Add new loss functions to `losses/custom_loss.py`:

```python
class YourCustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super(YourCustomLoss, self).__init__()
        
    def forward(self, *args):
        # Compute your custom loss
        return loss
```

## Research and References

This implementation is based on several influential papers in continual learning:

- [Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282)
- [Overcoming catastrophic forgetting with elastic weight consolidation (EWC)](https://arxiv.org/abs/1612.00796)
- [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)
- [A continual learning survey](https://arxiv.org/abs/1909.08383)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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

## Acknowledgments

This framework builds upon the extensive continual learning research community's work. Special thanks to all researchers advancing the field of lifelong learning.