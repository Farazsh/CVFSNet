"""DINOv3 ViT-S backbone and binary classification head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


SUPPORTED_BACKENDS = ("huggingface", "lightly")


def load_huggingface_backbone(
    pretrained_name: str,
    token: str | None = None,
    revision: str | None = None,
) -> nn.Module:
    """Load Meta's gated DINOv3 checkpoint through Transformers."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(pretrained_name, token=token, revision=revision)


def load_lightly_backbone(model_name: str = "dinov3/vits16") -> nn.Module:
    """Load a DINOv3 backbone and public weights through Lightly Train."""
    if not model_name.startswith("dinov3/"):
        raise ValueError("Lightly DINOv3 model names must start with 'dinov3/'.")
    try:
        from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError(
            "The 'lightly' backend requires the lightly-train package. Run `uv sync`."
        ) from error
    short_name = model_name.split("/", 1)[1]
    return DINOV3_PACKAGE.get_model(short_name, num_input_channels=3, load_weights=True)


class DINOv3BinaryClassifier(nn.Module):
    """Classify DSA temporal-channel images from CLS and mean patch features."""

    def __init__(
        self,
        pretrained_name: str,
        token: str | None = None,
        revision: str | None = None,
        dropout: float = 0.3,
        expected_num_register_tokens: int = 4,
        backend: str = "huggingface",
        lightly_name: str = "dinov3/vits16",
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backend = str(backend).lower()
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported DINOv3 backend '{backend}'; expected one of {SUPPORTED_BACKENDS}."
            )
        if backbone is None:
            if self.backend == "huggingface":
                backbone = load_huggingface_backbone(pretrained_name, token, revision)
            else:
                backbone = load_lightly_backbone(lightly_name)
        self.backbone = backbone
        if self.backend == "huggingface":
            hidden_size = int(self.backbone.config.hidden_size)
            self.patch_size = int(getattr(self.backbone.config, "patch_size", 16))
            register_tokens = getattr(
                self.backbone.config, "num_register_tokens", expected_num_register_tokens
            )
        else:
            hidden_size = int(self.backbone.embed_dim)
            self.patch_size = int(self.backbone.patch_size)
            register_tokens = getattr(
                self.backbone,
                "n_storage_tokens",
                getattr(self.backbone, "num_register_tokens", expected_num_register_tokens),
            )
        self.num_register_tokens = int(register_tokens)
        if self.num_register_tokens != int(expected_num_register_tokens):
            raise ValueError(
                f"Checkpoint has {self.num_register_tokens} register tokens; "
                f"expected {expected_num_register_tokens}."
            )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(
                f"Expected (B,3,H,W) temporal-channel inputs, got {tuple(pixel_values.shape)}."
            )
        if pixel_values.shape[-2] % self.patch_size or pixel_values.shape[-1] % self.patch_size:
            raise ValueError(
                f"Input height and width must be divisible by patch size {self.patch_size}."
            )
        if self.backend == "huggingface":
            cls_token, mean_patch = self._forward_huggingface(pixel_values)
        else:
            cls_token, mean_patch = self._forward_lightly(pixel_values)
        return self.classifier(torch.cat([cls_token, mean_patch], dim=1)).squeeze(1)

    def _forward_huggingface(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: Any = self.backbone(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        patch_start = 1 + self.num_register_tokens
        if hidden.ndim != 3 or hidden.shape[1] <= patch_start:
            raise ValueError("DINOv3 output does not contain the expected patch tokens.")
        return hidden[:, 0], hidden[:, patch_start:].mean(dim=1)

    def _forward_lightly(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: Any = self.backbone(pixel_values, is_training=True)
        try:
            cls_token = outputs["x_norm_clstoken"]
            patch_tokens = outputs["x_norm_patchtokens"]
        except (KeyError, TypeError) as error:
            raise ValueError("Lightly DINOv3 did not return normalized CLS/patch tokens.") from error
        if patch_tokens.ndim == 4:
            mean_patch = patch_tokens.flatten(2).mean(dim=2)
        elif patch_tokens.ndim == 3:
            mean_patch = patch_tokens.mean(dim=1)
        else:
            raise ValueError(f"Unexpected Lightly patch-token shape {tuple(patch_tokens.shape)}.")
        return cls_token, mean_patch

    def parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        """Separate backbone/head learning rates and norm/bias weight decay."""
        groups: list[dict[str, Any]] = []
        for module, lr in ((self.backbone, backbone_lr), (self.classifier, head_lr)):
            decay, no_decay = [], []
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                (no_decay if parameter.ndim == 1 else decay).append(parameter)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups
