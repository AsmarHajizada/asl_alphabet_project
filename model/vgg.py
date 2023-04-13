import torch
import torch.nn as nn
import torchvision.models as models


class vgg16(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Load a pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)  
        # Replace the last fully connected layer with a new one
        self.vgg16.classifier[6] = nn.Linear(4096, 1024)  
        self.relu = nn.ReLU(inplace=True)
        self.newfc = nn.Linear(1024, n_classes)


    def forward(self, image):
        # Get predictions from VGG16
        vgg_pred = self.vgg16(image)
        # Apply ReLU activation
        vgg_pred = self.relu(vgg_pred)
        vgg_pred = self.newfc(vgg_pred)

        return vgg_pred