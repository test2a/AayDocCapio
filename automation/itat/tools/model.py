"""6-head CNN for the ITAT captcha (alphanumeric; CPU friendly)."""
from __future__ import annotations

import torch.nn as nn

from charset import CAPTCHA_LEN, N_CLASSES, W, H

_FEAT_DIM = 64 * (H // 8) * (W // 8)


class CaptchaNet(nn.Module):
    def __init__(self, n_digits: int = CAPTCHA_LEN, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_digits = n_digits
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_FEAT_DIM, 192), nn.BatchNorm1d(192), nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.out = nn.ModuleList([nn.Linear(192, n_classes) for _ in range(n_digits)])

    def forward(self, x):
        f = self.head(self.features(x))
        return [o(f) for o in self.out]
