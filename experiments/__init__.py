"""
Experiments package for continual learning benchmarks.

This package contains benchmark scripts and utilities for evaluating
continual learning methods.
"""

from .continual_learning_benchmark import ContinualLearningBenchmark
from .cluster_aware_cl import ClusterAwareReplayFreeCL

__all__ = [
    'ContinualLearningBenchmark',
    'ClusterAwareReplayFreeCL'
]
