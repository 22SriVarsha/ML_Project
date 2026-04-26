import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class EncodeDepth(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(64, 128)

    def forward(self, depth):
        faetures_1 = self.layers(depth)
        faetures_1 = faetures_1.view(faetures_1.size(0), -1)
        faetures_1 = self.fc(faetures_1)
        return faetures_1


class RgbPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_rgb = resnet18(weights=ResNet18_Weights.DEFAULT)
        # get features from rgb model
        rft = self.model_rgb.fc.in_features
        self.model_rgb.fc = nn.Identity()
        self.depth_model = EncodeDepth()
        # sequential regressor
        self.regressor = nn.Sequential(
            nn.Linear(rft + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, rgb, depth):
        rgb_features = self.model_rgb(rgb)
        depth_features = self.depth_model(depth)
        combined_features = torch.cat([rgb_features, depth_features], dim=1)
        output = self.regressor(combined_features)
        return output