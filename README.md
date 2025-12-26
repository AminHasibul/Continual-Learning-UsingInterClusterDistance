# Continual Learning Project

A comprehensive implementation of continual learning algorithms and benchmarks for mitigating catastrophic forgetting in neural networks.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Supported Algorithms](#supported-algorithms)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Results](#results)
- [Usage Examples](#usage-examples)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [References](#references)

## 🎯 Overview

Continual learning (also known as lifelong learning or incremental learning) addresses the challenge of learning new tasks sequentially without forgetting previously learned knowledge. This repository provides implementations of state-of-the-art continual learning algorithms and comprehensive benchmarking tools.

### The Catastrophic Forgetting Problem

When neural networks learn new tasks, they often suffer from **catastrophic forgetting** - the tendency to completely forget previously learned tasks. This project implements various strategies to mitigate this problem:

- **Regularization-based approaches**: Add constraints to preserve important parameters
- **Rehearsal-based methods**: Store and replay examples from previous tasks  
- **Architecture-based solutions**: Dynamically expand or modify network structure
- **Meta-learning approaches**: Learn how to learn new tasks efficiently

## ✨ Key Features

- 🧠 **Multiple Algorithms**: Implementation of 10+ continual learning algorithms
- 📊 **Comprehensive Benchmarks**: Standard benchmarks (Split-CIFAR, Split-MNIST, etc.)
- 🔄 **Easy Experimentation**: Modular design for testing different scenarios
- 📈 **Detailed Metrics**: Forward/backward transfer, forgetting measures
- 🛠️ **Flexible Configuration**: YAML-based experiment configuration
- 📱 **Visualization Tools**: Built-in plotting and analysis utilities
- 🚀 **GPU Support**: Efficient training with CUDA acceleration

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AminHasibul/Continual-Learning-.git
   cd Continual-Learning-
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode:**
   ```bash
   pip install -e .
   ```

## ⚡ Quick Start

### Basic Example

```python
from continual_learning import ContinualLearner
from continual_learning.algorithms import EWC
from continual_learning.datasets import SplitCIFAR10

# Initialize dataset
dataset = SplitCIFAR10(n_tasks=5)

# Setup continual learner with EWC algorithm
learner = ContinualLearner(
    algorithm=EWC(lambda_reg=400),
    backbone='resnet18',
    optimizer='adam',
    lr=0.001
)

# Train on sequence of tasks
results = learner.train(dataset)

# Evaluate performance
accuracy = learner.evaluate(dataset)
print(f"Final accuracy: {accuracy:.2f}%")
```

### Command Line Interface

```bash
# Train EWC on Split-CIFAR10
python main.py --algorithm ewc --dataset split_cifar10 --tasks 5

# Run comprehensive benchmark
python benchmark.py --config configs/benchmark_config.yaml

# Visualize results
python plot_results.py --experiment_dir results/ewc_split_cifar10/
```

## 📁 Project Structure

```
Continual-Learning-/
├── continual_learning/          # Main package
│   ├── algorithms/              # Continual learning algorithms
│   │   ├── ewc.py              # Elastic Weight Consolidation
│   │   ├── lwf.py              # Learning without Forgetting
│   │   ├── gem.py              # Gradient Episodic Memory
│   │   ├── agem.py             # Averaged GEM
│   │   ├── packnet.py          # PackNet
│   │   └── ...
│   ├── datasets/               # Dataset implementations
│   │   ├── split_datasets.py   # Split MNIST/CIFAR variants
│   │   ├── permuted_mnist.py   # Permuted MNIST
│   │   └── continuum.py        # Task sequence utilities
│   ├── models/                 # Neural network architectures
│   │   ├── resnet.py          # ResNet variants
│   │   ├── mlp.py             # Multi-layer perceptrons
│   │   └── cnn.py             # Convolutional networks
│   ├── metrics/               # Evaluation metrics
│   │   ├── accuracy.py        # Classification accuracy
│   │   ├── forgetting.py      # Forgetting measures
│   │   └── transfer.py        # Transfer learning metrics
│   └── utils/                 # Utility functions
├── configs/                   # Configuration files
│   ├── algorithms/           # Algorithm-specific configs
│   ├── datasets/            # Dataset configurations
│   └── experiments/         # Experiment setups
├── experiments/             # Experiment scripts
├── results/                # Output directory
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## 🧠 Supported Algorithms

### Regularization-Based
- **EWC** (Elastic Weight Consolidation) - Protects important weights for previous tasks
- **SI** (Synaptic Intelligence) - Online importance estimation
- **LwF** (Learning without Forgetting) - Knowledge distillation approach
- **MAS** (Memory Aware Synapses) - Importance-based parameter protection

### Rehearsal-Based  
- **GEM** (Gradient Episodic Memory) - Constrained optimization with episodic memory
- **A-GEM** (Averaged GEM) - Efficient approximation of GEM
- **ER** (Experience Replay) - Simple replay buffer approach
- **DER** (Dark Experience Replay) - Enhanced replay with dark knowledge

### Architecture-Based
- **PackNet** - Pruning-based approach with task-specific subnetworks
- **HAT** (Hard Attention to Task) - Attention-based parameter selection
- **ProgressiveNet** - Progressive network expansion

### Meta-Learning
- **MAML** (Model-Agnostic Meta-Learning) - Learn good initialization
- **Reptile** - First-order meta-learning algorithm

## 📊 Datasets

### Benchmark Datasets
- **Split-MNIST**: MNIST digits split into 5 binary classification tasks
- **Split-CIFAR10**: CIFAR-10 classes split into 5 tasks (2 classes each)
- **Split-CIFAR100**: CIFAR-100 classes split into 10/20 tasks
- **Permuted-MNIST**: MNIST with different pixel permutations per task
- **Rotated-MNIST**: MNIST with different rotations per task

### Real-World Scenarios
- **Core50**: Video object recognition benchmark
- **CUB-200**: Fine-grained bird classification
- **Stream-51**: Streaming classification benchmark

## ⚙️ Configuration

Experiments are configured using YAML files. Example configuration:

```yaml
# config/ewc_experiment.yaml
algorithm:
  name: "ewc"
  params:
    lambda_reg: 400.0
    online: false

dataset:
  name: "split_cifar10"
  n_tasks: 5
  batch_size: 32

model:
  backbone: "resnet18"
  pretrained: false

training:
  epochs_per_task: 50
  optimizer: "adam"
  learning_rate: 0.001
  scheduler: "step"

evaluation:
  metrics: ["accuracy", "forgetting", "forward_transfer"]
  save_results: true
```

## 📈 Results

### Performance on Split-CIFAR10

| Algorithm | Final Accuracy | Avg. Forgetting | Forward Transfer |
|-----------|---------------|-----------------|------------------|
| Fine-tuning | 19.8% | 68.4% | -2.1% |
| EWC | 47.2% | 23.1% | 1.8% |
| LwF | 45.6% | 25.3% | 0.9% |
| GEM | 52.1% | 18.7% | 3.2% |
| A-GEM | 49.8% | 20.4% | 2.1% |
| PackNet | 56.3% | 12.1% | 4.7% |

### Key Metrics

- **Final Accuracy**: Performance on all tasks after learning sequence
- **Average Forgetting**: How much performance drops on previous tasks  
- **Forward Transfer**: Knowledge transfer to future tasks
- **Backward Transfer**: Knowledge transfer from future to past tasks

## 💡 Usage Examples

### Custom Algorithm Implementation

```python
from continual_learning.algorithms.base import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize algorithm-specific parameters
    
    def train_task(self, task_data):
        # Implement task-specific training
        pass
    
    def after_task(self, task_id):
        # Post-task processing (e.g., importance estimation)
        pass
```

### Custom Dataset

```python
from continual_learning.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize dataset
    
    def get_task(self, task_id):
        # Return (train_loader, test_loader) for task
        pass
```

### Hyperparameter Tuning

```python
from continual_learning.utils import grid_search

# Define parameter grid
param_grid = {
    'lambda_reg': [100, 400, 1000],
    'learning_rate': [0.01, 0.001, 0.0001]
}

# Run grid search
best_params = grid_search(
    algorithm_class=EWC,
    dataset=SplitCIFAR10(),
    param_grid=param_grid,
    metric='final_accuracy'
)
```

## 🔬 Benchmarks

### Running Benchmarks

```bash
# Run all algorithms on Split-CIFAR10
python benchmark.py --dataset split_cifar10 --algorithms ewc lwf gem agem

# Custom benchmark with specific configuration
python benchmark.py --config configs/my_benchmark.yaml

# Multi-seed evaluation for statistical significance
python benchmark.py --seeds 5 --dataset split_mnist --algorithm ewc
```

### Adding Custom Benchmarks

Create a benchmark configuration file:

```yaml
# configs/benchmarks/my_benchmark.yaml
datasets:
  - name: "split_cifar10"
    n_tasks: 5
  - name: "split_mnist"  
    n_tasks: 5

algorithms:
  - name: "ewc"
    params: {lambda_reg: 400}
  - name: "gem"
    params: {memory_size: 500}

evaluation:
  seeds: [42, 123, 456]
  metrics: ["accuracy", "forgetting"]
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest tests/`
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all public functions
- Run `black` and `isort` for formatting

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{continual_learning_2024,
  title={Continual Learning: A Comprehensive Implementation},
  author={Md Hasibul Amin},
  year={2024},
  url={https://github.com/AminHasibul/Continual-Learning-}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

### Key Papers

1. **Elastic Weight Consolidation (EWC)**
   - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)

2. **Learning without Forgetting (LwF)**  
   - Li & Hoiem, "Learning without forgetting" (TPAMI 2017)

3. **Gradient Episodic Memory (GEM)**
   - Lopez-Paz & Ranzato, "Gradient episodic memory for continual learning" (NIPS 2017)

4. **Averaged GEM (A-GEM)**
   - Chaudhry et al., "Efficient lifelong learning with A-GEM" (ICLR 2019)

5. **PackNet**
   - Mallya & Lazebnik, "PackNet: Adding multiple tasks to a single network by iterative pruning" (CVPR 2018)

### Surveys and Resources

- **Continual Learning Literature**: [ContinualAI Wiki](https://wiki.continualai.org/)
- **Survey Papers**: 
  - De Lange et al., "A continual learning survey: Defying forgetting in classification tasks" (TPAMI 2021)
  - Parisi et al., "Continual lifelong learning with neural networks: A review" (Neural Networks 2019)

### Related Repositories

- [Avalanche](https://github.com/ContinualAI/avalanche) - End-to-end library for continual learning
- [Continuum](https://github.com/Continvvm/continuum) - Clean continual learning datasets
- [Mammoth](https://github.com/aimagelab/mammoth) - Continual learning framework

---

**Contact**: [AminHasibul](https://github.com/AminHasibul)

For questions, issues, or suggestions, please open an issue or reach out via email.
