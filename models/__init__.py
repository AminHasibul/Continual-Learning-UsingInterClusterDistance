"""
Models package for continual learning experiments.

This package contains custom neural network architectures designed
for continual learning tasks.
"""

from .custom_cnn import CustomCNN
from .custom_resnet import CustomResNet, create_resnet18, create_resnet34

__all__ = ['CustomCNN', 'CustomResNet', 'create_resnet18', 'create_resnet34']
