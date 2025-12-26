"""
Custom ResNet Architecture for Continual Learning

This module implements a custom ResNet-style architecture designed for
continual learning scenarios with techniques to prevent catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic ResNet block with two convolutional layers and skip connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the first convolution (default: 1)
        downsample (nn.Module): Downsampling layer for skip connection (default: None)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class CustomResNet(nn.Module):
    """
    Custom ResNet architecture for continual learning.
    
    This architecture can be modified to incorporate continual learning
    techniques such as:
    - Progressive neural networks
    - Learning without forgetting
    - Elastic weight consolidation
    - iCaRL (Incremental Classifier and Representation Learning)
    
    Args:
        block (nn.Module): Type of residual block (BasicBlock or Bottleneck)
        layers (list): Number of blocks in each layer
        num_classes (int): Number of output classes (default: 10)
        input_channels (int): Number of input channels (default: 3 for RGB)
    """
    
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], 
                 num_classes=10, input_channels=3):
        super(CustomResNet, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a residual layer with multiple blocks.
        
        Args:
            block (nn.Module): Type of residual block
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract feature representations before the final classification layer.
        
        This is useful for continual learning techniques that operate on
        feature representations.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature representations
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_params_by_layer(self):
        """
        Get parameters grouped by layer for importance-based continual learning.
        
        Returns:
            dict: Dictionary mapping layer names to parameters
        """
        params = {
            'conv1': list(self.conv1.parameters()) + list(self.bn1.parameters()),
            'layer1': list(self.layer1.parameters()),
            'layer2': list(self.layer2.parameters()),
            'layer3': list(self.layer3.parameters()),
            'layer4': list(self.layer4.parameters()),
            'fc': list(self.fc.parameters())
        }
        return params


def create_resnet18(num_classes=10, input_channels=3):
    """
    Create a ResNet-18 model for continual learning.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels
        
    Returns:
        CustomResNet: ResNet-18 model
    """
    return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)


def create_resnet34(num_classes=10, input_channels=3):
    """
    Create a ResNet-34 model for continual learning.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels
        
    Returns:
        CustomResNet: ResNet-34 model
    """
    return CustomResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels)


if __name__ == "__main__":
    # Test ResNet-18
    print("Testing ResNet-18:")
    model = create_resnet18(num_classes=10, input_channels=3)
    print(model)
    
    # Test with random input
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4, 32x32 RGB images
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
