"""
Runtime-safe model architectures shared by Foundry training and EchoZero inference.
Exists because both apps need the same checkpoint-compatible network definitions.
Used by trainers during export and by runtime loaders during inference.
"""

from __future__ import annotations

import torch


class SimpleCnnRuntimeModel(torch.nn.Module):
    """Small mel-spectrogram CNN for runtime classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 8 * 8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CrnnRuntimeModel(torch.nn.Module):
    """Convolutional recurrent runtime model for transient classification."""

    def __init__(self, num_classes: int, mel_bins: int, hidden_size: int = 96):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 1)),
        )
        reduced_mels = max(1, mel_bins // 4)
        self.rnn = torch.nn.GRU(
            input_size=32 * reduced_mels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.permute(0, 3, 1, 2).contiguous().flatten(start_dim=2)
        seq, _ = self.rnn(feat)
        pooled = seq.mean(dim=1)
        return self.classifier(pooled)
