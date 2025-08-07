# models.py
"""Modèles IA pour drawgen"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SketchClassifier(nn.Module):
    """Classificateur de sketches basé sur ResNet"""
    
    def __init__(self, num_classes: int = 50, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Chargement du backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            # Adaptation pour images en niveaux de gris
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone non supporté: {backbone}")
        
        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
        return self.classifier(features)

class DepthEstimator(nn.Module):
    """Estimateur de profondeur U-Net simplifié"""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1, features: int = 64):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encodeur
        self.enc1 = self._conv_block(input_channels, features)
        self.enc2 = self._conv_block(features, features * 2)
        self.enc3 = self._conv_block(features * 2, features * 4)
        
        # Décodeur
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.dec2 = self._conv_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.dec1 = self._conv_block(features * 2, features)
        
        # Sortie
        self.final = nn.Conv2d(features, output_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encodage avec skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Décodage avec skip connections
        dec2 = self.upconv2(enc3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Sortie avec sigmoid pour normaliser entre 0 et 1
        output = torch.sigmoid(self.final(dec1))
        return output