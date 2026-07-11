"""VideoMAE backbone + CORN ordinal head for coronal mTICI grading.

Design (see PLAN.md):
    * Backbone: ``VideoMAEModel`` (Base), pretrained on Kinetics-400
      (``MCG-NJU/videomae-base-finetuned-kinetics``).
    * Input adapter -> VideoMAE / ImageNet normalization: the data pipeline
      yields a grayscale clip ``(B, 1, T, H, W)`` at an arbitrary (very small)
      intensity scale. VideoMAE has NO input BatchNorm and adds fixed sinusoidal
      position embeddings, so it is highly sensitive to input scale. We therefore
      (a) per-clip min-max rescale to ``[0, 1]``, (b) replicate 1->3 channels,
      (c) apply ImageNet mean/std -- exactly the stats the pretrained checkpoint
      expects. This restores unit-ish variance and is what fixes the earlier
      collapse.
    * Head: mean-pool the token sequence -> LayerNorm -> dropout -> Linear ->
      ``K - 1`` CORN logits.
    * ``freeze_backbone``: when true, the backbone is frozen (linear-probe style);
      only the head trains.

``forward`` returns the raw ``(B, K-1)`` CORN logits.
"""

from __future__ import annotations

from typing import Any, List, Mapping

import torch
import torch.nn as nn

from transformers import VideoMAEModel

# Standard ImageNet statistics used by VideoMAEImageProcessor.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoMAEOrdinal(nn.Module):
    """VideoMAE encoder with a CORN ordinal-regression head."""

    def __init__(
        self,
        num_classes: int,
        pretrained_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        hidden_dropout_prob: float | None = None,
        normalize: str = "imagenet",
        expected_num_frames: int = 16,
        expected_image_size: int = 224,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal regression.")
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.normalize = normalize
        self.expected_num_frames = expected_num_frames
        self.expected_image_size = expected_image_size

        # Optionally inject a light dropout into the transformer (HF VideoMAE has
        # no stochastic depth; this is a mild regularization substitute).
        overrides: dict = {}
        if hidden_dropout_prob is not None:
            overrides["hidden_dropout_prob"] = float(hidden_dropout_prob)
            overrides["attention_probs_dropout_prob"] = float(hidden_dropout_prob)
        self.backbone = VideoMAEModel.from_pretrained(pretrained_name, **overrides)
        hidden_size = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes - 1),
        )

        # ImageNet normalization buffers (broadcast over B, T, C, H, W).
        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False
        )

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):  # noqa: D401
        """Keep a frozen backbone in eval mode even when the module trains."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _to_pixel_values(self, clip: torch.Tensor) -> torch.Tensor:
        """``(B, 1, T, H, W)`` grayscale -> ImageNet-normalized ``(B, T, 3, H, W)``."""
        if clip.ndim != 5:
            raise ValueError(f"Expected a 5D clip (B,C,T,H,W), got {tuple(clip.shape)}.")
        b, c, t, h, w = clip.shape
        if c == 1:
            clip = clip.expand(b, 3, t, h, w)
        elif c != 3:
            raise ValueError(f"Unexpected channel count {c}; expected 1 or 3.")

        # (B, C, T, H, W) -> (B, T, C, H, W)
        pixel_values = clip.permute(0, 2, 1, 3, 4).contiguous()

        if self.normalize == "imagenet":
            # Per-clip min-max to [0, 1] so the arbitrary DSA intensity scale is
            # mapped into the range ImageNet stats assume.
            flat = pixel_values.reshape(b, -1)
            min_v = flat.min(dim=1).values.view(b, 1, 1, 1, 1)
            max_v = flat.max(dim=1).values.view(b, 1, 1, 1, 1)
            pixel_values = (pixel_values - min_v) / (max_v - min_v + 1e-6)
            pixel_values = (pixel_values - self.imagenet_mean) / self.imagenet_std
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize mode '{self.normalize}'.")
        return pixel_values

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        pixel_values = self._to_pixel_values(clip)
        outputs = self.backbone(pixel_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)


def build_videomae_ordinal(config: Mapping[str, Any], num_classes: int) -> VideoMAEOrdinal:
    """Build the model from a config ``params`` mapping."""
    params = dict(config.get("params", {}))
    return VideoMAEOrdinal(
        num_classes=num_classes,
        pretrained_name=params.get(
            "pretrained_name", "MCG-NJU/videomae-base-finetuned-kinetics"
        ),
        dropout=float(params.get("dropout", 0.5)),
        freeze_backbone=bool(params.get("freeze_backbone", False)),
        hidden_dropout_prob=params.get("hidden_dropout_prob"),
        normalize=str(params.get("normalize", "imagenet")),
        expected_num_frames=int(params.get("expected_num_frames", 16)),
        expected_image_size=int(params.get("expected_image_size", 224)),
    )


class VideoMAEBinary(nn.Module):
    """VideoMAE encoder with a single-logit binary classification head."""

    def __init__(
        self,
        pretrained_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        hidden_dropout_prob: float | None = None,
        normalize: str = "imagenet",
        expected_num_frames: int = 16,
        expected_image_size: int = 224,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.normalize = normalize
        self.expected_num_frames = expected_num_frames
        self.expected_image_size = expected_image_size

        overrides: dict = {}
        if hidden_dropout_prob is not None:
            overrides["hidden_dropout_prob"] = float(hidden_dropout_prob)
            overrides["attention_probs_dropout_prob"] = float(hidden_dropout_prob)
        self.backbone = VideoMAEModel.from_pretrained(pretrained_name, **overrides)
        hidden_size = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False
        )

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _to_pixel_values(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"Expected a 5D clip (B,C,T,H,W), got {tuple(clip.shape)}.")
        b, c, t, h, w = clip.shape
        if c == 1:
            clip = clip.expand(b, 3, t, h, w)
        elif c != 3:
            raise ValueError(f"Unexpected channel count {c}; expected 1 or 3.")
        pixel_values = clip.permute(0, 2, 1, 3, 4).contiguous()
        if self.normalize == "imagenet":
            flat = pixel_values.reshape(b, -1)
            min_v = flat.min(dim=1).values.view(b, 1, 1, 1, 1)
            max_v = flat.max(dim=1).values.view(b, 1, 1, 1, 1)
            pixel_values = (pixel_values - min_v) / (max_v - min_v + 1e-6)
            pixel_values = (pixel_values - self.imagenet_mean) / self.imagenet_std
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize mode '{self.normalize}'.")
        return pixel_values

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled backbone features for one clip."""
        pixel_values = self._to_pixel_values(clip)
        outputs = self.backbone(pixel_values)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(clip))


class VideoMAEBinaryDual(nn.Module):
    """Shared-backbone dual-view binary classifier (late fusion)."""

    def __init__(
        self,
        pretrained_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        hidden_dropout_prob: float | None = None,
        normalize: str = "imagenet",
        expected_num_frames: int = 16,
        expected_image_size: int = 224,
    ) -> None:
        super().__init__()
        self.encoder = VideoMAEBinary(
            pretrained_name=pretrained_name,
            dropout=0.0,
            freeze_backbone=freeze_backbone,
            hidden_dropout_prob=hidden_dropout_prob,
            normalize=normalize,
            expected_num_frames=expected_num_frames,
            expected_image_size=expected_image_size,
        )
        hidden_size = self.encoder.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(mode)
        return self

    def forward(self, clips: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(clips) != 2:
            raise ValueError(f"Expected exactly 2 view clips, got {len(clips)}.")
        ap_feat = self.encoder.encode(clips[0])
        sag_feat = self.encoder.encode(clips[1])
        return self.head(torch.cat([ap_feat, sag_feat], dim=1))


def binary_pseudologits(logit: torch.Tensor) -> torch.Tensor:
    """Map a single logit ``(B, 1)`` to two-class scores ``(B, 2)`` for metrics."""
    if logit.ndim == 1:
        logit = logit.unsqueeze(1)
    return torch.cat([-logit, logit], dim=1)


def build_videomae_binary(config: Mapping[str, Any]) -> nn.Module:
    """Build a binary VideoMAE model from config ``params``."""
    params = dict(config.get("params", {}))
    common = dict(
        pretrained_name=params.get(
            "pretrained_name", "MCG-NJU/videomae-base-finetuned-kinetics"
        ),
        dropout=float(params.get("dropout", 0.5)),
        freeze_backbone=bool(params.get("freeze_backbone", False)),
        hidden_dropout_prob=params.get("hidden_dropout_prob"),
        normalize=str(params.get("normalize", "imagenet")),
        expected_num_frames=int(params.get("expected_num_frames", 16)),
        expected_image_size=int(params.get("expected_image_size", 224)),
    )
    name = str(config.get("name", "videomae_binary"))
    if name == "videomae_binary_dual":
        return VideoMAEBinaryDual(**common)
    return VideoMAEBinary(**common)


def layerwise_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> List[dict]:
    """Layer-wise LR decay groups (VideoMAE / BEiT convention, decay 0.75).

    Deeper layers (closer to the head) get a larger LR; the patch embedding gets
    the smallest. Bias / 1-D (norm) params get no weight decay. Frozen params are
    skipped. See VideoMAE FINETUNE.md (``--layer_decay 0.75``).
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None and hasattr(model, "encoder"):
        backbone = model.encoder.backbone
    if backbone is None:
        raise AttributeError("Model has no backbone for layer-wise LR decay.")
    num_layers = backbone.config.num_hidden_layers + 1  # + head group

    def layer_id(name: str) -> int:
        for prefix in ("backbone.", "encoder.backbone."):
            if name.startswith(prefix + "embeddings"):
                return 0
            marker = prefix + "encoder.layer."
            if name.startswith(marker):
                return int(name.split(marker)[1].split(".")[0]) + 1
        return num_layers  # head, final norm, anything else

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid = layer_id(name)
        scale = layer_decay ** (num_layers - lid)
        no_decay = param.ndim == 1
        key = (lid, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(param)
    return list(groups.values())
