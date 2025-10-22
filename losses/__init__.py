"""
Losses package for continual learning experiments.

This package contains custom loss functions designed for continual learning,
including techniques to prevent catastrophic forgetting.
"""

from .custom_loss import (
    KnowledgeDistillationLoss,
    ElasticWeightConsolidationLoss,
    LwFLoss,
    iCaRLLoss,
    compute_fisher_information
)

__all__ = [
    'KnowledgeDistillationLoss',
    'ElasticWeightConsolidationLoss', 
    'LwFLoss',
    'iCaRLLoss',
    'compute_fisher_information'
]
