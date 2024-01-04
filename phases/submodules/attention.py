import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, scale_dot_product=False):
        super(SpatialAttention, self).__init__()
        self.scale_dot_product = scale_dot_product
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights
        weights = self.conv1(x)
        weights = self.sigmoid(weights)
        
        if self.scale_dot_product:
            weights = weights / torch.sqrt(torch.sum(weights**2, dim=(2, 3), keepdim=True) + 1e-8)

        # Apply attention to the input
        x = x * weights
        return x

class resnet18attention(nn.Module):
    def __init__(self, num_classes, attention_channels=512, use_mask=False, scale_dot_product=False, pretrained=True):
        super(resnet18attention, self).__init__()
        # Load pre-trained ResNet
        self.resnet = resnet18(pretrained=pretrained)
        # Remove the original fully connected layer
        self.resnet.fc = nn.Identity()
        # Add attention mechanism
        self.attention = SpatialAttention(attention_channels, scale_dot_product)
        # Add new fully connected layer
        self.fc = nn.Linear(attention_channels, num_classes)
        # Set mask flag.
        self.use_mask = use_mask

    def forward(self, x, mask=None):
        # Hard attention
        if self.use_mask and mask is not None:
            x = x * mask
        # Get ResNet features
        features = self.resnet(x).reshape(-1, 512, 1, 1)
        # Apply soft attention
        attended_features = self.attention(features)
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(attended_features, (1, 1))
        # Flatten for fully connected layer
        flattened_features = torch.flatten(pooled_features, 1)
        # Fully connected layer
        output = self.fc(flattened_features)
        return output

if __name__ == '__main__':
    # Example usage
    num_classes = 3  # Change this based on your classification task
    model = resnet18attention(num_classes, use_mask=True, scale_dot_product=True)
    print(model)