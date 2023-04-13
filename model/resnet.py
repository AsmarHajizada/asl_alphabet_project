import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Load a pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)  
        # Replace the last fully connected layer with a new one
        self.resnet18.fc = nn.Linear(512, 1024)  
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1024, n_classes) 

    def forward(self, image):
        # Get the predictions from pretrained ResNet18
        pred = self.resnet18(image)
        # Apply ReLU activation 
        pred = self.relu(pred)
        # Apply the new fully connected layer
        pred = self.fc(pred)

        return pred
