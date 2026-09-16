from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 6
    feature_dim: int = 128
    lstm_hidden_size: int = 128
    lstm_layers: int = 1
    dropout: float = 0.3
    frame_chunk_size: int = 128


class FrameEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        blocks = []
        channels = (3, 16, 32, 64, 128)
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(nn.AvgPool2d(4), *blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(frames))


class CNNBiLSTM(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.frame_encoder = FrameEncoder(self.config.feature_dim)
        self.temporal_encoder = nn.LSTM(
            input_size=self.config.feature_dim,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.lstm_hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.num_classes),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5 or videos.shape[2] != 3:
            raise ValueError("expected videos with shape [B, T, 3, H, W]")
        batch_size, frames = videos.shape[:2]
        flat = videos.reshape(batch_size * frames, *videos.shape[2:])
        features = torch.cat([
            self.frame_encoder(chunk)
            for chunk in flat.split(self.config.frame_chunk_size)
        ], dim=0)
        sequence = features.reshape(batch_size, frames, self.config.feature_dim)
        _, (hidden, _) = self.temporal_encoder(sequence)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(representation)

    def metadata(self) -> dict[str, object]:
        return asdict(self.config)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
