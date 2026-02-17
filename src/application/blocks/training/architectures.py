"""
Model Architectures for Audio Classification

Provides all neural network architectures used for audio classification training.
Includes custom CNN, ResNet/EfficientNet transfer learning, RNN, Transformer,
Wav2Vec2, and Ensemble models.

All architectures implement a common interface:
    __init__(num_classes: int, config: dict)
    forward(x: Tensor) -> Tensor

For binary mode, num_classes=1 and the output is a single logit (apply sigmoid).
For multiclass, num_classes=N and output is N logits (apply softmax).
"""
from typing import Any, Dict, List, Optional
from functools import partial

import sys
from src.utils.message import Log

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    _mod = sys.modules.get(__name__)
    if _mod is not None and not getattr(_mod, "_logged_torchvision", False):
        Log.debug("torchvision not available - ResNet/EfficientNet transfer learning disabled")
        setattr(_mod, "_logged_torchvision", True)

try:
    from transformers import Wav2Vec2Model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    _mod = sys.modules.get(__name__)
    if _mod is not None and not getattr(_mod, "_logged_transformers", False):
        Log.debug("transformers not available - Wav2Vec2 models disabled")
        setattr(_mod, "_logged_transformers", True)


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Learns to weight frequency bands by importance, which is especially
    useful for audio where certain frequency ranges are diagnostic
    (e.g., kicks are dominated by 40-100 Hz).

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution -- same accuracy as standard conv at ~1/9th
    the parameters. Used in MobileNet and EfficientNet.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


# ---------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------

class AudioClassifierBase(nn.Module):
    """Base class for audio classifiers with common functionality."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 1


# ---------------------------------------------------------------------------
# CNN Classifier
# ---------------------------------------------------------------------------

class CNNClassifier(AudioClassifierBase):
    """
    Configurable CNN for audio spectrogram classification.

    Supports optional Squeeze-and-Excitation attention and depthwise separable
    convolutions for efficiency.
    """

    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__(num_classes)

        self.dropout_rate = config.get("dropout_rate", 0.4)
        num_conv_layers = config.get("num_conv_layers", 4)
        base_channels = config.get("base_channels", 32)
        use_se = config.get("use_se_blocks", False)

        # Calculate the minimum input dimension (time axis) to determine
        # how many MaxPool2d(2,2) layers we can safely use. Each pool
        # halves the spatial dims, so we need min_dim >= 2 after pooling.
        n_mels = config.get("n_mels", 128)
        max_length = config.get("max_length", 22050)
        hop_length = config.get("hop_length", 512)
        min_dim = min(n_mels, max_length // hop_length + 1)
        import math
        max_safe_pools = max(1, int(math.log2(max(min_dim, 2))) - 1)

        # Build convolutional layers dynamically
        layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = base_channels * (2 ** min(i, 4))  # Cap channel growth at 16x
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if use_se:
                layers.append(SEBlock(out_channels))
            # Only add MaxPool if spatial dims can handle it
            if i < max_safe_pools:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # AdaptiveAvgPool guarantees a fixed (1x1) spatial output regardless
        # of how many pooling layers were used above.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(self.dropout_rate)
        fc_hidden = config.get("fc_hidden_size", 512)
        self.fc1 = nn.Linear(in_channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# ResNet Classifier (Transfer Learning)
# ---------------------------------------------------------------------------

class ResNetClassifier(AudioClassifierBase):
    """
    ResNet-based audio classifier using transfer learning from ImageNet.

    Modifies the first conv layer to accept 1-channel mel spectrogram input
    by averaging the pre-trained 3-channel weights. This is the single most
    proven approach for audio spectrogram classification (used by Google,
    Spotify, and most audio ML teams).

    Supports resnet18, resnet34, resnet50.
    """

    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__(num_classes)

        if not HAS_TORCHVISION:
            raise ImportError(
                "torchvision is required for ResNet models. "
                "Install with: pip install torchvision"
            )

        backbone_name = config.get("model_type", "resnet18").lower()
        pretrained = config.get("pretrained_backbone", True)
        dropout_rate = config.get("dropout_rate", 0.4)

        # Load backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone_name == "resnet18":
            backbone = tv_models.resnet18(weights=weights)
        elif backbone_name == "resnet34":
            backbone = tv_models.resnet34(weights=weights)
        elif backbone_name == "resnet50":
            backbone = tv_models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown ResNet variant: {backbone_name}")

        # Adapt first conv layer: 3-channel RGB -> 1-channel spectrogram
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained:
            # Average RGB weights to create single-channel weights
            with torch.no_grad():
                self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        # Copy everything except conv1 and fc
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # New classification head
        in_features = backbone.fc.in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# EfficientNet Classifier (Transfer Learning)
# ---------------------------------------------------------------------------

class EfficientNetClassifier(AudioClassifierBase):
    """
    EfficientNet-based audio classifier using transfer learning from ImageNet.

    State-of-the-art accuracy/efficiency tradeoff. Same 1-channel adaptation
    as ResNet. Supports efficientnet_b0, efficientnet_b1, efficientnet_b2.
    """

    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__(num_classes)

        if not HAS_TORCHVISION:
            raise ImportError(
                "torchvision is required for EfficientNet models. "
                "Install with: pip install torchvision"
            )

        backbone_name = config.get("model_type", "efficientnet_b0").lower()
        pretrained = config.get("pretrained_backbone", True)
        dropout_rate = config.get("dropout_rate", 0.4)

        # Load backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone_name == "efficientnet_b0":
            backbone = tv_models.efficientnet_b0(weights=weights)
        elif backbone_name == "efficientnet_b1":
            backbone = tv_models.efficientnet_b1(weights=weights)
        elif backbone_name == "efficientnet_b2":
            backbone = tv_models.efficientnet_b2(weights=weights)
        else:
            raise ValueError(f"Unknown EfficientNet variant: {backbone_name}")

        # Adapt first conv layer: 3 channels -> 1 channel
        original_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # New classification head
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# RNN Classifier
# ---------------------------------------------------------------------------

class RNNClassifier(AudioClassifierBase):
    """
    RNN-based audio classifier using spectrogram time sequences.
    Supports LSTM, GRU, and vanilla RNN with optional attention.
    """

    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__(num_classes)

        input_size = config.get("rnn_input_size", 128)
        hidden_size = config.get("rnn_hidden_size", 256)
        num_layers = config.get("rnn_num_layers", 2)
        rnn_type = config.get("rnn_type", "LSTM").upper()
        bidirectional = config.get("rnn_bidirectional", True)
        dropout_rate = config.get("dropout_rate", 0.3)

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}.get(rnn_type, nn.LSTM)
        self.rnn = rnn_cls(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        directions = 2 if bidirectional else 1
        self.use_attention = config.get("use_attention", False)
        if self.use_attention:
            self.attention = nn.Linear(hidden_size * directions, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, mel_bins, time_steps)
        x = x.squeeze(1).transpose(1, 2)  # (batch, time_steps, mel_bins)

        rnn_out, _ = self.rnn(x)

        if self.use_attention:
            weights = torch.softmax(self.attention(rnn_out), dim=1)
            context = torch.sum(rnn_out * weights, dim=1)
        else:
            context = rnn_out[:, -1, :]

        x = self.dropout(context)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Transformer Classifier
# ---------------------------------------------------------------------------

class TransformerClassifier(AudioClassifierBase):
    """Transformer encoder for audio spectrogram classification."""

    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__(num_classes)

        input_size = config.get("transformer_input_size", 128)
        d_model = config.get("transformer_d_model", 256)
        nhead = config.get("transformer_nhead", 8)
        num_layers = config.get("transformer_num_layers", 4)
        dropout_rate = config.get("dropout_rate", 0.1)

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_rate, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, mel_bins, time_steps)
        x = x.squeeze(1).transpose(1, 2)  # (batch, time_steps, mel_bins)

        x = self.input_proj(x)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)

        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Mean pooling

        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Wav2Vec2 Classifier
# ---------------------------------------------------------------------------

class Wav2Vec2Classifier(AudioClassifierBase):
    """
    Wav2Vec2-based classifier using pre-trained audio representations.
    Requires raw audio waveform input (not spectrograms).
    """

    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__(num_classes)

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for Wav2Vec2 models")

        model_name = config.get("wav2vec2_model", "facebook/wav2vec2-base")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if config.get("freeze_wav2vec2", True):
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        hidden_size = self.wav2vec2.config.hidden_size
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.3))
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: raw audio waveform (batch, samples)
        outputs = self.wav2vec2(x)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        pooled = torch.mean(hidden_states, dim=1)  # Mean pooling
        x = self.dropout(pooled)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Ensemble Classifier
# ---------------------------------------------------------------------------

class EnsembleClassifier(AudioClassifierBase):
    """Ensemble of multiple classifiers with weighted averaging or meta-classifier."""

    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__(num_classes)

        self.models = nn.ModuleList()
        self.weights = config.get("ensemble_weights", [])

        for model_config in config.get("ensemble_models", []):
            model = create_classifier(num_classes, model_config)
            self.models.append(model)

        if not self.weights:
            self.weights = [1.0 / max(len(self.models), 1)] * len(self.models)

        self.use_meta = config.get("use_meta_classifier", False)
        if self.use_meta and self.models:
            self.meta_fc = nn.Linear(len(self.models) * num_classes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.models:
            raise ValueError("Ensemble has no models")

        predictions = [m(x) for m in self.models]

        if self.use_meta:
            combined = torch.cat(predictions, dim=1)
            return self.meta_fc(combined)

        return sum(w * p for w, p in zip(self.weights, predictions))


# ---------------------------------------------------------------------------
# Architecture Registry and Factory
# ---------------------------------------------------------------------------

# Map of architecture name -> callable(num_classes, config) -> model
ARCHITECTURE_REGISTRY: Dict[str, type] = {
    "cnn": CNNClassifier,
    "resnet18": ResNetClassifier,
    "resnet34": ResNetClassifier,
    "resnet50": ResNetClassifier,
    "efficientnet_b0": EfficientNetClassifier,
    "efficientnet_b1": EfficientNetClassifier,
    "efficientnet_b2": EfficientNetClassifier,
    "rnn": RNNClassifier,
    "lstm": RNNClassifier,
    "gru": RNNClassifier,
    "transformer": TransformerClassifier,
    "wav2vec2": Wav2Vec2Classifier,
    "ensemble": EnsembleClassifier,
}


def create_classifier(num_classes: int, config: Dict[str, Any]) -> AudioClassifierBase:
    """
    Create a classifier model from configuration.

    Args:
        num_classes: Number of output classes (1 for binary, N for multiclass)
        config: Configuration dictionary (or TrainingConfig.to_dict())

    Returns:
        AudioClassifierBase instance

    Raises:
        ValueError: If model_type is unknown
    """
    model_type = config.get("model_type", "cnn").lower()

    cls = ARCHITECTURE_REGISTRY.get(model_type)
    if cls is None:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")

    return cls(num_classes, config)


# Backward-compatible alias
class AudioClassifierFactory:
    """Factory for creating audio classifiers (backward compatibility)."""

    @staticmethod
    def create_classifier(num_classes: int, config: Dict[str, Any]) -> AudioClassifierBase:
        return create_classifier(num_classes, config)
