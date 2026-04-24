import torch
import torch.nn as nn
import timm


class XceptionDeepfake(nn.Module):
    def __init__(self):
        super(XceptionDeepfake, self).__init__()

        self.backbone = timm.create_model(
            "xception",
            pretrained=False,   # IMPORTANT: weights come from checkpoint
            num_classes=0
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
