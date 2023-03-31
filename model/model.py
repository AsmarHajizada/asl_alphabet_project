import torch
import torch.nn as nn
from dataset.dataset_retrieval import custom_dataset

    
class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Third convolution layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Classifier layers
        self.classifier_layer1 = nn.Linear(784, 128)
        self.classifier_layer2 = nn.Linear(128, n_classes)

        self.flatten = nn.Flatten()

        # Output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.classifier_layer1(x)
        x = self.relu(x)
        x = self.classifier_layer2(x)
        
        x = self.softmax(x)
        
        print(x.shape)
        return x
        
