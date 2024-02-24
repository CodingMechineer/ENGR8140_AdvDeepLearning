import torch
from torch import nn
from torch.nn import functional as F

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid

from torchvision.models import resnet50
from ptflops import get_model_complexity_info

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm import tqdm
import torch.optim as optim

import os

# # Accelerate parts
# from accelerate import Accelerator, notebook_launcher # main interface, distributed launcher
# from accelerate.utils import set_seed # reproducability across devices


batch_size = 16 #256
maxEpoch = 30

def get_dataloaders(batch_size):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    cf10_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train = CIFAR10(root='./data', train=True, download=True, transform=cf10_transforms)
    test = CIFAR10(root='./data', train=False, download=True, transform=cf10_transforms)
    
    ##################################################################################
    print(train.classes)
    
    torch.manual_seed(42)
    val_size = 5000
    train_size = len(train) - val_size

    ##################################################################################
    
    # 'random_split' is a PyTorch function that randomly splits a dataset into non-overlapping new datasets
    # The lengths of the splits are provided as a list: [train_size, val_size]
    train_ds, val_ds = random_split(train, [train_size, val_size])
    len(train_ds), len(val_ds)
    
    ##################################################################################
    
    # Create a DataLoader for the training dataset
    # shuffle=True will shuffle the dataset before each epoch
    # num_workers=2 will use two subprocesses for data loading
    # pin_memory=True will copy Tensors into CUDA pinned memory before returning them
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train.classes


class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SepConv, self).__init__()
        # Depthwise convolution followed by BatchNorm and ReLU
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Pointwise convolution followed by BatchNorm without ReLU
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )      

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.fc_expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)

    def forward(self, x):
        se_weight = self.avg_pool(x)
        se_weight = F.relu(self.fc_reduce(se_weight))
        se_weight = torch.sigmoid(self.fc_expand(se_weight))
        return x * se_weight


class MBConv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor=3):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = in_channels * expansion_factor

        # NEW!
        # if not self.use_residual:
        #     self.identity_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.se_block = SEBlock(mid_channels, reduced_channels=mid_channels // 4) # Why through 4??
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # identity = x if self.use_residual else None
        # identity = self.identity_conv(x) if not self.use_residual else x

        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se_block(x)
        x = self.project_conv(x)
        # if self.use_residual:
        #    x += identity
        return x


class MBConv6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor=6):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = in_channels * expansion_factor

        # New!
        # if not self.use_residual:
        #     self.identity_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=2, groups=mid_channels, bias=False), # padding=2 to keep the same size ???
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # identity = x if self.use_residual else None
        # identity = self.identity_conv(x) if not self.use_residual else x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        # if self.use_residual:
        #      x += identity
        
        return x
    
class MBConv6_SE(MBConv6):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor=6):
        super().__init__(in_channels, out_channels, kernel_size, stride, expansion_factor=6)
        self.se_block = SEBlock(out_channels, reduced_channels=out_channels // 4) # Why through 4??

    def forward(self, x):
        x = super().forward(x)
        x = self.se_block(x)
        return x
    
# ----------------------------------------------------------------------------------------------
# class MBConv6(nn.Module):
#     def _init_(self, in_channels, out_channels, kernel_size, stride, expansion_factor=6):
    
# class MBConv3(nn.Module):
#     def _init_(self, in_channels, out_channels, kernel_size, stride, expansion_factor=3):
# ----------------------------------------------------------------------------------------------

# Define the MnasNet-A1 model
class MnasNetA1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Initial 3x3 convolution
        self.initial_conv = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        # Initial separable convolution
        self.sep_conv = SepConv(32, 16, 3)
        # Sequence of MBConv blocks
        self.blocks = nn.Sequential(
            MBConv6(16, 24, kernel_size=3, stride = 2),
            MBConv6(24, 24, kernel_size=3, stride = 1),
            
            MBConv3(24, 40, kernel_size=3, stride = 2),
            MBConv3(40, 40, kernel_size=3, stride = 1),
            MBConv3(40, 40, kernel_size=3, stride = 1),
            
            MBConv6(40, 80, kernel_size=3, stride = 2),
            MBConv6(80, 80, kernel_size=3, stride = 1),
            MBConv6(80, 80, kernel_size=3, stride = 1),
            MBConv6(80, 80, kernel_size=3, stride = 1),

            MBConv6_SE(80, 112, kernel_size=3, stride = 1),
            MBConv6_SE(112, 112, kernel_size=3, stride = 1),

            MBConv6_SE(112, 160, kernel_size=5, stride = 2),
            MBConv6_SE(160, 160, kernel_size=5, stride = 2),
            MBConv6_SE(160, 160, kernel_size=5, stride = 2),

            MBConv6(160, 320, kernel_size=3, stride = 1),
        )

        # Global average pooling and classifier as mentioned above
        # self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final separable convolution
        self.final_conv = SepConv(320, 320, 3)

        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        # Apply initial convolution
        x = self.initial_conv(x)
        # Apply initial separable convolution
        x = self.sep_conv(x)
        # Apply MBConv blocks
        x = self.blocks(x)
        
        # Apply global pooling to the output of the last layer
        #x = self.global_pool(x)

        # Pass the flattened output through the final classifier layer
        # This will output the raw scores (logits) for each class
        x = self.final_conv(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)

        # Flatten the output from the global pooling layer
        # The '1' argument means that we start flattening from the second dimension (0-indexed)
        # So if the input shape was (batch_size, num_channels, height, width), the output shape will be (batch_size, num_channels*height*width)
        # x = torch.flatten(x, 1)
        x = self.fc(x)
        print(x.shape)

        # Return the logits
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Build the DataLoaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size)

    # Initialize the model, loss function, optimizer, and scheduler
    model = MnasNetA1(len(classes)).to(device)  # Adjust the number of classes based on your dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)  # Reduce learning rate by a factor of 0.1 every 10 epochs

    # Very important to use DataParallel to use multiple GPUs
    model = nn.DataParallel(model)

    # Plot accuracy over epochs
    plt.figure(figsize=(7, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')

    # List to store accuracy for each epoch
    accuracy_list = []

    # Training loop
    for epoch in range(maxEpoch):  # Loop over the dataset multiple times
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Use tqdm to create a progress bar for the training loop
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{maxEpoch}", unit="batch") as progress_bar:
            for i, data in enumerate(progress_bar):
                # Get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data[0].to(accelerator.device), data[1].to(accelerator.device)
                inputs, labels = data[0].to(device), data[1].to(device)

                ########################################################################################
                # Training the model

                # Zero the parameter gradients
                optimizer.zero_grad()

                print(inputs.shape)  # Add this line before your forward pass
                
                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step() 

                ########################################################################################   

                # Update statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # Update progress bar
                progress_bar.set_postfix(loss=total_loss / (i+1), accuracy=correct_predictions / total_samples)

        # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        train_accuracy = correct_predictions / total_samples

        loss = total_loss / total_samples

        # Optionally, you can print or log other metrics at the end of each epoch
        print(f"Epoch {epoch+1}/{maxEpoch}, Loss: {loss}, Accuracy: {train_accuracy}, 'Validation Accuracy: {accuracy:.2f}%'")

        plt.plot(range(1, epoch+2), accuracy_list)
        plt.draw()
        
    print('Finished Training')


if __name__ == '__main__':
    main()