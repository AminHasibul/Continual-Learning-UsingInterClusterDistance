"""
Continual Learning Benchmark

This module provides a framework for benchmarking continual learning methods
on various datasets and task sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import time


class ContinualLearningBenchmark:
    """
    Benchmark framework for continual learning experiments.
    
    This class provides utilities for:
    - Sequential task training
    - Evaluation on all seen tasks
    - Computing forgetting metrics
    - Tracking performance over time
    
    Args:
        model (nn.Module): Neural network model
        device (str): Device to run experiments on ('cpu' or 'cuda')
        num_tasks (int): Number of tasks in the sequence
    """
    
    def __init__(self, model, device='cpu', num_tasks=5):
        self.model = model.to(device)
        self.device = device
        self.num_tasks = num_tasks
        self.task_history = []
        self.performance_matrix = np.zeros((num_tasks, num_tasks))
        
    def train_task(self, 
                   task_id: int,
                   train_loader: DataLoader,
                   epochs: int = 10,
                   learning_rate: float = 0.001,
                   criterion: Optional[nn.Module] = None,
                   use_regularization: bool = False,
                   verbose: bool = True):
        """
        Train the model on a single task.
        
        Args:
            task_id (int): ID of the current task
            train_loader (DataLoader): Training data loader
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            criterion (nn.Module): Loss function (default: CrossEntropyLoss)
            use_regularization (bool): Whether to use continual learning regularization
            verbose (bool): Whether to print training progress
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Compute loss
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, labels)
                else:
                    # Custom loss that might need additional arguments
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            if verbose:
                print(f'Task {task_id}, Epoch {epoch+1}/{epochs}: '
                      f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Store task information
        self.task_history.append({
            'task_id': task_id,
            'model_state': copy.deepcopy(self.model.state_dict())
        })
    
    def evaluate_task(self, 
                      task_id: int,
                      test_loader: DataLoader,
                      verbose: bool = True) -> Tuple[float, float]:
        """
        Evaluate the model on a specific task.
        
        Args:
            task_id (int): ID of the task to evaluate
            test_loader (DataLoader): Test data loader
            verbose (bool): Whether to print results
            
        Returns:
            Tuple[float, float]: (accuracy, average_loss)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        if verbose:
            print(f'Task {task_id} Evaluation: '
                  f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return accuracy, avg_loss
    
    def evaluate_all_tasks(self,
                          task_loaders: List[DataLoader],
                          current_task_id: int,
                          verbose: bool = True) -> Dict[int, float]:
        """
        Evaluate the model on all tasks seen so far.
        
        Args:
            task_loaders (List[DataLoader]): List of test data loaders for each task
            current_task_id (int): ID of the current task
            verbose (bool): Whether to print results
            
        Returns:
            Dict[int, float]: Dictionary mapping task IDs to accuracies
        """
        results = {}
        
        if verbose:
            print(f"\n=== Evaluation after Task {current_task_id} ===")
        
        for task_id in range(current_task_id + 1):
            accuracy, _ = self.evaluate_task(
                task_id, 
                task_loaders[task_id],
                verbose=verbose
            )
            results[task_id] = accuracy
            self.performance_matrix[current_task_id, task_id] = accuracy
        
        return results
    
    def compute_metrics(self, current_task_id: int) -> Dict[str, float]:
        """
        Compute continual learning metrics.
        
        Args:
            current_task_id (int): ID of the current task
            
        Returns:
            Dict[str, float]: Dictionary of metrics including:
                - average_accuracy: Average accuracy on all tasks
                - forgetting: Average forgetting measure
                - backward_transfer: Backward transfer measure
        """
        metrics = {}
        
        # Average accuracy
        accuracies = self.performance_matrix[current_task_id, :current_task_id+1]
        metrics['average_accuracy'] = np.mean(accuracies)
        
        # Forgetting measure
        if current_task_id > 0:
            forgetting = []
            for t in range(current_task_id):
                max_acc = np.max(self.performance_matrix[:current_task_id+1, t])
                current_acc = self.performance_matrix[current_task_id, t]
                forgetting.append(max_acc - current_acc)
            metrics['forgetting'] = np.mean(forgetting)
        else:
            metrics['forgetting'] = 0.0
        
        # Backward transfer
        if current_task_id > 0:
            backward_transfer = []
            for t in range(current_task_id):
                initial_acc = self.performance_matrix[t, t]
                current_acc = self.performance_matrix[current_task_id, t]
                backward_transfer.append(current_acc - initial_acc)
            metrics['backward_transfer'] = np.mean(backward_transfer)
        else:
            metrics['backward_transfer'] = 0.0
        
        return metrics
    
    def run_benchmark(self,
                     train_loaders: List[DataLoader],
                     test_loaders: List[DataLoader],
                     epochs_per_task: int = 10,
                     learning_rate: float = 0.001,
                     save_path: Optional[str] = None) -> Dict:
        """
        Run complete continual learning benchmark.
        
        Args:
            train_loaders (List[DataLoader]): Training data loaders for each task
            test_loaders (List[DataLoader]): Test data loaders for each task
            epochs_per_task (int): Number of epochs per task
            learning_rate (float): Learning rate
            save_path (str): Path to save results (optional)
            
        Returns:
            Dict: Benchmark results including performance matrix and metrics
        """
        print("=" * 60)
        print("Starting Continual Learning Benchmark")
        print("=" * 60)
        
        start_time = time.time()
        
        for task_id in range(self.num_tasks):
            print(f"\n{'='*60}")
            print(f"Training Task {task_id}")
            print(f"{'='*60}")
            
            # Train on current task
            self.train_task(
                task_id=task_id,
                train_loader=train_loaders[task_id],
                epochs=epochs_per_task,
                learning_rate=learning_rate
            )
            
            # Evaluate on all tasks seen so far
            self.evaluate_all_tasks(test_loaders, task_id)
            
            # Compute and display metrics
            metrics = self.compute_metrics(task_id)
            print(f"\nMetrics after Task {task_id}:")
            print(f"  Average Accuracy: {metrics['average_accuracy']:.2f}%")
            print(f"  Forgetting: {metrics['forgetting']:.2f}%")
            print(f"  Backward Transfer: {metrics['backward_transfer']:.2f}%")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("Benchmark Complete")
        print("=" * 60)
        print(f"Total Time: {total_time:.2f} seconds")
        
        # Prepare results
        results = {
            'performance_matrix': self.performance_matrix,
            'final_metrics': self.compute_metrics(self.num_tasks - 1),
            'total_time': total_time
        }
        
        # Save results if path provided
        if save_path:
            torch.save(results, save_path)
            print(f"Results saved to {save_path}")
        
        return results
    
    def print_performance_matrix(self):
        """Print the performance matrix in a readable format."""
        print("\nPerformance Matrix:")
        print("Rows: After training on task i")
        print("Columns: Accuracy on task j")
        print("-" * 60)
        
        # Header
        header = "      " + "  ".join([f"T{i:2d}" for i in range(self.num_tasks)])
        print(header)
        print("-" * 60)
        
        # Matrix rows
        for i in range(self.num_tasks):
            row = f"T{i:2d} |"
            for j in range(self.num_tasks):
                if j <= i:
                    row += f" {self.performance_matrix[i, j]:5.1f}"
                else:
                    row += "     -"
            print(row)


def create_task_split(dataset, num_tasks, task_type='class_incremental'):
    """
    Split a dataset into multiple tasks.
    
    Args:
        dataset: PyTorch dataset
        num_tasks (int): Number of tasks
        task_type (str): Type of split ('class_incremental' or 'domain_incremental')
        
    Returns:
        List[Subset]: List of dataset subsets for each task
    """
    if task_type == 'class_incremental':
        # Split by classes
        num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else 10
        classes_per_task = num_classes // num_tasks
        
        task_datasets = []
        for task_id in range(num_tasks):
            start_class = task_id * classes_per_task
            end_class = start_class + classes_per_task
            
            # Get indices for this task's classes
            indices = [i for i, (_, label) in enumerate(dataset) 
                      if start_class <= label < end_class]
            
            task_datasets.append(Subset(dataset, indices))
        
        return task_datasets
    else:
        raise NotImplementedError(f"Task type {task_type} not implemented")


if __name__ == "__main__":
    print("Continual Learning Benchmark Framework")
    print("=" * 60)
    print("\nThis module provides tools for benchmarking continual learning methods.")
    print("\nExample usage:")
    print("""
    # Create model
    from models import CustomCNN
    model = CustomCNN(num_classes=10)
    
    # Create benchmark
    benchmark = ContinualLearningBenchmark(model, device='cuda', num_tasks=5)
    
    # Prepare task loaders
    # train_loaders = [loader1, loader2, ...]
    # test_loaders = [test1, test2, ...]
    
    # Run benchmark
    results = benchmark.run_benchmark(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_per_task=10
    )
    
    # Print results
    benchmark.print_performance_matrix()
    """)
