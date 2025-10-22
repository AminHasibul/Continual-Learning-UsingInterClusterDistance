"""
Custom Loss Functions for Continual Learning

This module implements various loss functions designed to prevent
catastrophic forgetting in continual learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for continual learning.
    
    This loss helps transfer knowledge from a previous model (teacher)
    to a new model (student) while learning new tasks.
    
    Args:
        temperature (float): Temperature for softening probability distributions
        alpha (float): Weight balancing distillation and classification loss
    """
    
    def __init__(self, temperature=2.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits (torch.Tensor): Logits from student model
            teacher_logits (torch.Tensor): Logits from teacher model
            labels (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Classification loss
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(
            student_soft, 
            teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * distillation_loss
        
        return total_loss


class ElasticWeightConsolidationLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC) Loss.
    
    EWC prevents catastrophic forgetting by constraining important parameters
    to stay close to their values from previous tasks.
    
    Args:
        lambda_ewc (float): Regularization strength for EWC
    """
    
    def __init__(self, lambda_ewc=1000.0):
        super(ElasticWeightConsolidationLoss, self).__init__()
        self.lambda_ewc = lambda_ewc
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels, model, fisher_dict, optimal_params):
        """
        Compute EWC loss.
        
        Args:
            logits (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels
            model (nn.Module): Current model
            fisher_dict (dict): Fisher information matrix for each parameter
            optimal_params (dict): Optimal parameters from previous task
            
        Returns:
            torch.Tensor: Combined loss with EWC regularization
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)
        
        # EWC regularization
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fisher_dict:
                fisher = fisher_dict[name]
                optimal_param = optimal_params[name]
                ewc_loss += (fisher * (param - optimal_param) ** 2).sum()
        
        total_loss = ce_loss + (self.lambda_ewc / 2) * ewc_loss
        
        return total_loss


class LwFLoss(nn.Module):
    """
    Learning without Forgetting (LwF) Loss.
    
    LwF uses knowledge distillation to preserve performance on old tasks
    while learning new tasks.
    
    Args:
        temperature (float): Temperature for knowledge distillation
        alpha (float): Weight for old task preservation
    """
    
    def __init__(self, temperature=2.0, alpha=0.5):
        super(LwFLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, new_logits, old_logits, labels, old_task_idx=None):
        """
        Compute LwF loss.
        
        Args:
            new_logits (torch.Tensor): Logits from current model
            old_logits (torch.Tensor): Logits from previous model (detached)
            labels (torch.Tensor): Ground truth labels for new task
            old_task_idx (list): Indices of old task classes (optional)
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Loss for new task
        new_task_loss = self.ce_loss(new_logits, labels)
        
        # Distillation loss for old tasks
        if old_logits is not None and old_task_idx is not None:
            # Extract logits for old classes
            new_old_logits = new_logits[:, old_task_idx]
            old_old_logits = old_logits[:, old_task_idx]
            
            # Compute distillation loss
            new_soft = F.log_softmax(new_old_logits / self.temperature, dim=1)
            old_soft = F.softmax(old_old_logits / self.temperature, dim=1)
            distillation_loss = F.kl_div(
                new_soft,
                old_soft,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            total_loss = self.alpha * new_task_loss + (1 - self.alpha) * distillation_loss
        else:
            total_loss = new_task_loss
        
        return total_loss


def compute_fisher_information(model, data_loader, device='cpu', num_samples=None):
    """
    Compute Fisher Information Matrix for EWC.
    
    The Fisher Information Matrix estimates the importance of each parameter
    for the current task.
    
    Args:
        model (nn.Module): Neural network model
        data_loader (DataLoader): Data loader for the current task
        device (str): Device to run computations on
        num_samples (int): Maximum number of samples to use (None for all)
        
    Returns:
        dict: Dictionary mapping parameter names to Fisher information values
        dict: Dictionary of current optimal parameters
    """
    model.eval()
    fisher_dict = {}
    optimal_params = {}
    
    # Initialize Fisher information
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
            optimal_params[name] = param.data.clone()
    
    # Compute Fisher information
    count = 0
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        if num_samples is not None and count >= num_samples:
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        model.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        count += inputs.size(0)
    
    # Normalize by number of samples
    for name in fisher_dict:
        fisher_dict[name] /= count
    
    return fisher_dict, optimal_params


class iCaRLLoss(nn.Module):
    """
    iCaRL (Incremental Classifier and Representation Learning) Loss.
    
    Combines classification loss with distillation loss for continual learning.
    
    Args:
        temperature (float): Temperature for distillation
        alpha (float): Balance between classification and distillation
    """
    
    def __init__(self, temperature=2.0, alpha=0.5):
        super(iCaRLLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, logits, labels, old_logits=None):
        """
        Compute iCaRL loss.
        
        Args:
            logits (torch.Tensor): Current model logits
            labels (torch.Tensor): Ground truth labels
            old_logits (torch.Tensor): Previous model logits (optional)
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Binary cross-entropy for classification
        num_classes = logits.size(1)
        targets = torch.zeros_like(logits)
        targets.scatter_(1, labels.unsqueeze(1), 1)
        
        # Sigmoid cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Distillation loss if old model exists
        if old_logits is not None:
            # Use sigmoid for distillation in iCaRL
            current_probs = torch.sigmoid(logits)
            old_probs = torch.sigmoid(old_logits)
            
            distillation_loss = F.binary_cross_entropy(
                current_probs,
                old_probs,
                reduction='mean'
            )
            
            total_loss = self.alpha * bce_loss + (1 - self.alpha) * distillation_loss
        else:
            total_loss = bce_loss
        
        return total_loss


if __name__ == "__main__":
    # Test Knowledge Distillation Loss
    print("Testing Knowledge Distillation Loss:")
    kd_loss = KnowledgeDistillationLoss(temperature=2.0, alpha=0.5)
    student_logits = torch.randn(32, 10)
    teacher_logits = torch.randn(32, 10)
    labels = torch.randint(0, 10, (32,))
    loss = kd_loss(student_logits, teacher_logits, labels)
    print(f"KD Loss: {loss.item():.4f}")
    
    # Test EWC Loss
    print("\nTesting EWC Loss:")
    ewc_loss_fn = ElasticWeightConsolidationLoss(lambda_ewc=1000.0)
    # Note: EWC requires a model, fisher dict, and optimal params
    print("EWC Loss requires model and Fisher information (see usage in experiments)")
    
    # Test LwF Loss
    print("\nTesting LwF Loss:")
    lwf_loss = LwFLoss(temperature=2.0, alpha=0.5)
    new_logits = torch.randn(32, 10)
    old_logits = torch.randn(32, 10)
    labels = torch.randint(0, 10, (32,))
    loss = lwf_loss(new_logits, old_logits, labels, old_task_idx=list(range(5)))
    print(f"LwF Loss: {loss.item():.4f}")
    
    # Test iCaRL Loss
    print("\nTesting iCaRL Loss:")
    icarl_loss = iCaRLLoss(temperature=2.0, alpha=0.5)
    logits = torch.randn(32, 10)
    labels = torch.randint(0, 10, (32,))
    loss = icarl_loss(logits, labels)
    print(f"iCaRL Loss: {loss.item():.4f}")
