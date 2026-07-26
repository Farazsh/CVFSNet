"""VideoMAE backbone + CORN ordinal head for coronal mTICI grading.

Design (see mae_ordinal/docs/ORDINAL_EXPERIMENT_PLAN.md):
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

import os
from pathlib import Path
from typing import Any, List, Mapping

import torch
import torch.nn as nn

from transformers import VideoMAEModel

# Standard ImageNet statistics used by VideoMAEImageProcessor.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_repo_dotenv() -> None:
    """Load repo-root ``.env`` so ``HF_TOKEN`` is available for Hub downloads."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def resolve_hf_token(token_env: str = "HF_TOKEN") -> str | None:
    """Return the Hugging Face token from the environment / repo ``.env``."""
    _load_repo_dotenv()
    token = os.environ.get(token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        token = token.strip().strip('"').strip("'")
    return token or None


def _hf_auth_kwargs(token_env: str = "HF_TOKEN") -> dict[str, Any]:
    token = resolve_hf_token(token_env)
    return {"token": token} if token else {}


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
        self.backbone = VideoMAEModel.from_pretrained(
            pretrained_name, **overrides, **_hf_auth_kwargs()
        )
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
        self.backbone = VideoMAEModel.from_pretrained(
            pretrained_name, **overrides, **_hf_auth_kwargs()
        )
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


def _videomaev2_model_config(config) -> dict:
    model_cfg = getattr(config, "model_config", None)
    if model_cfg is None:
        raise AttributeError("VideoMAE2 config is missing model_config.")
    return dict(model_cfg)


def adapt_videomaev2_num_frames(backbone: nn.Module, num_frames: int) -> None:
    """Rebuild tubelet count / sin-cos ``pos_embed`` for a new temporal length.

    Conv3d patch embedding weights are frame-count agnostic; only the number of
    tokens and positional table change with ``num_frames // tubelet_size``.
    """
    num_frames = int(num_frames)
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}.")
    model_cfg = _videomaev2_model_config(backbone.config)
    tubelet = int(model_cfg.get("tubelet_size", 2))
    if num_frames % tubelet != 0:
        raise ValueError(
            f"num_frames={num_frames} must be divisible by tubelet_size={tubelet}."
        )
    vit = backbone.model
    patch = vit.patch_embed
    img_size = patch.img_size
    if isinstance(img_size, (tuple, list)):
        h, w = int(img_size[0]), int(img_size[1])
    else:
        h = w = int(img_size)
    patch_size = patch.patch_size
    if isinstance(patch_size, (tuple, list)):
        ph, pw = int(patch_size[0]), int(patch_size[1])
    else:
        ph = pw = int(patch_size)
    num_spatial = (h // ph) * (w // pw)
    num_temporal = num_frames // tubelet
    num_patches = num_spatial * num_temporal

    # Prefer the Hub module's own sinusoid helper when available.
    module = type(vit).__module__
    try:
        import importlib

        modeling = importlib.import_module(module)
        get_table = getattr(modeling, "get_sinusoid_encoding_table")
        pos_embed = get_table(num_patches, vit.embed_dim)
    except Exception:
        # Fallback: truncated/padded copy of the existing table (rare).
        old = vit.pos_embed.detach()
        pos_embed = old.new_zeros(1, num_patches, old.shape[-1])
        copy_n = min(num_patches, old.shape[1])
        pos_embed[:, :copy_n] = old[:, :copy_n]

    vit.pos_embed = pos_embed
    patch.num_patches = num_patches
    if hasattr(patch, "num_frames"):
        patch.num_frames = num_frames
    model_cfg["num_frames"] = num_frames
    backbone.config.model_config = model_cfg
    backbone.model_config = model_cfg


def load_videomaev2_backbone(
    pretrained_name: str = "OpenGVLab/VideoMAEv2-Base",
    trust_remote_code: bool = True,
    token_env: str = "HF_TOKEN",
    num_frames: int = 16,
) -> nn.Module:
    """Load OpenGVLab VideoMAE2 with a transformers-version-safe path.

    Newer ``transformers`` + Hub remote code can fail on ``from_pretrained``
    (missing ``all_tied_weights_keys``) or leave sinusoidal ``pos_embed`` on the
    meta device. Load via ``from_config`` and apply safetensors weights instead.

    ``num_frames`` may differ from the Hub default (16). Patch-projection weights
    are reused; sin-cos positional embeddings are regenerated for the new length.

    Uses ``HF_TOKEN`` from the repo ``.env`` (or the environment) for authenticated
    Hub downloads when available.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    auth = _hf_auth_kwargs(token_env)
    config = AutoConfig.from_pretrained(
        pretrained_name, trust_remote_code=trust_remote_code, **auth
    )
    model_cfg = _videomaev2_model_config(config)
    pretrained_frames = int(model_cfg.get("num_frames", 16))
    model_cfg["num_frames"] = int(num_frames)
    config.model_config = model_cfg

    backbone = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
    if not hasattr(backbone, "all_tied_weights_keys"):
        type(backbone).all_tied_weights_keys = {}
    weights_path = hf_hub_download(
        pretrained_name, "model.safetensors", **auth
    )
    state = load_file(weights_path)
    # Drop tensors whose shapes no longer match (typically ``pos_embed`` when
    # num_frames != pretrained_frames). Fresh sin-cos embeddings from from_config
    # are kept for the requested temporal length.
    model_state = backbone.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    skipped = sorted(set(state) - set(filtered))
    if unexpected:
        raise RuntimeError(
            f"VideoMAE2 weight load had unexpected keys for '{pretrained_name}': "
            f"{unexpected[:8]}"
        )
    # Missing keys are OK when only pos_embed was skipped / freshly initialized.
    del missing
    if int(num_frames) != pretrained_frames:
        adapt_videomaev2_num_frames(backbone, int(num_frames))
    # Ensure non-parameter buffers (sin-cos pos_embed) are materialized.
    for name, buffer in backbone.named_buffers():
        if getattr(buffer, "is_meta", False):
            raise RuntimeError(f"VideoMAE2 buffer '{name}' is still on the meta device.")
    if skipped and int(num_frames) == pretrained_frames:
        raise RuntimeError(
            f"VideoMAE2 skipped unexpected weight keys at native frame count: {skipped[:8]}"
        )
    return backbone


class VideoMAEv2Binary(nn.Module):
    """VideoMAE2 encoder with a single-logit binary classification head.

    Expects pipeline clips ``(B, 1|3, T, H, W)`` and feeds the backbone
    ``(B, 3, T, H, W)`` after ImageNet-style intensity adaptation.
    """

    def __init__(
        self,
        pretrained_name: str = "OpenGVLab/VideoMAEv2-Base",
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        hidden_dropout_prob: float | None = None,
        normalize: str = "imagenet",
        expected_num_frames: int = 16,
        expected_image_size: int = 224,
        trust_remote_code: bool = True,
        token_env: str = "HF_TOKEN",
    ) -> None:
        super().__init__()
        del hidden_dropout_prob  # VideoMAE2 uses drop_path in its own config.
        self.freeze_backbone = freeze_backbone
        self.normalize = normalize
        self.expected_num_frames = expected_num_frames
        self.expected_image_size = expected_image_size

        self.backbone = load_videomaev2_backbone(
            pretrained_name,
            trust_remote_code=trust_remote_code,
            token_env=token_env,
            num_frames=int(expected_num_frames),
        )
        model_cfg = getattr(self.backbone, "model_config", None)
        if isinstance(model_cfg, dict):
            hidden_size = int(model_cfg.get("embed_dim", 768))
            self._num_layers = int(model_cfg.get("depth", 12))
        else:
            vit = getattr(self.backbone, "model", None)
            hidden_size = int(getattr(vit, "embed_dim", 768))
            self._num_layers = int(getattr(vit, "get_num_layers", lambda: 12)())
        self.hidden_size = hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Broadcast over (B, C, T, H, W).
        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False
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
        """``(B, 1|3, T, H, W)`` -> ImageNet-normalized ``(B, 3, T, H, W)``."""
        if clip.ndim != 5:
            raise ValueError(f"Expected a 5D clip (B,C,T,H,W), got {tuple(clip.shape)}.")
        b, c, t, h, w = clip.shape
        if t != self.expected_num_frames:
            raise ValueError(
                f"Expected {self.expected_num_frames} frames, got T={t}. "
                "Set data.pipeline_overrides.data.num_frames to match "
                "model.params.expected_num_frames."
            )
        if c == 1:
            clip = clip.expand(b, 3, t, h, w)
        elif c != 3:
            raise ValueError(f"Unexpected channel count {c}; expected 1 or 3.")
        pixel_values = clip.contiguous()
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
        if hasattr(self.backbone, "extract_features"):
            return self.backbone.extract_features(pixel_values)
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state.mean(dim=1)
        return outputs

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(clip))


class VideoMAEv2BinaryDual(nn.Module):
    """Shared VideoMAE2 backbone dual-view binary classifier (late fusion)."""

    def __init__(
        self,
        pretrained_name: str = "OpenGVLab/VideoMAEv2-Base",
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        hidden_dropout_prob: float | None = None,
        normalize: str = "imagenet",
        expected_num_frames: int = 16,
        expected_image_size: int = 224,
        trust_remote_code: bool = True,
        token_env: str = "HF_TOKEN",
    ) -> None:
        super().__init__()
        self.encoder = VideoMAEv2Binary(
            pretrained_name=pretrained_name,
            dropout=0.0,
            freeze_backbone=freeze_backbone,
            hidden_dropout_prob=hidden_dropout_prob,
            normalize=normalize,
            expected_num_frames=expected_num_frames,
            expected_image_size=expected_image_size,
            trust_remote_code=trust_remote_code,
            token_env=token_env,
        )
        hidden_size = int(self.encoder.hidden_size)
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


def build_videomae_binary(config: Mapping[str, Any]) -> nn.Module:
    """Build a binary VideoMAE / VideoMAE2 model from config ``params``."""
    params = dict(config.get("params", {}))
    name = str(config.get("name", "videomae_binary"))
    is_v2 = name.startswith("videomaev2_")
    common = dict(
        pretrained_name=params.get(
            "pretrained_name",
            "OpenGVLab/VideoMAEv2-Base"
            if is_v2
            else "MCG-NJU/videomae-base-finetuned-kinetics",
        ),
        dropout=float(params.get("dropout", 0.5)),
        freeze_backbone=bool(params.get("freeze_backbone", False)),
        hidden_dropout_prob=params.get("hidden_dropout_prob"),
        normalize=str(params.get("normalize", "imagenet")),
        expected_num_frames=int(params.get("expected_num_frames", 16)),
        expected_image_size=int(params.get("expected_image_size", 224)),
    )
    if name == "videomaev2_binary_dual":
        return VideoMAEv2BinaryDual(
            **common,
            trust_remote_code=bool(params.get("trust_remote_code", True)),
            token_env=str(params.get("token_env", "HF_TOKEN")),
        )
    if name == "videomaev2_binary":
        return VideoMAEv2Binary(
            **common,
            trust_remote_code=bool(params.get("trust_remote_code", True)),
            token_env=str(params.get("token_env", "HF_TOKEN")),
        )
    if name == "videomae_binary_dual":
        return VideoMAEBinaryDual(**common)
    if name == "videomae_binary":
        return VideoMAEBinary(**common)
    raise ValueError(
        f"Unknown binary model name '{name}'. Expected videomae_binary, "
        "videomae_binary_dual, videomaev2_binary, or videomaev2_binary_dual."
    )


def _backbone_num_layers(backbone: nn.Module) -> int:
    if hasattr(backbone, "_num_layers"):
        return int(backbone._num_layers)
    config = getattr(backbone, "config", None)
    if config is not None and hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    model_cfg = getattr(backbone, "model_config", None)
    if isinstance(model_cfg, dict) and "depth" in model_cfg:
        return int(model_cfg["depth"])
    vit = getattr(backbone, "model", None)
    if vit is not None and hasattr(vit, "get_num_layers"):
        return int(vit.get_num_layers())
    if vit is not None and hasattr(vit, "blocks"):
        return len(vit.blocks)
    raise AttributeError("Could not determine backbone depth for layer-wise LR decay.")


def layerwise_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> List[dict]:
    """Layer-wise LR decay groups (VideoMAE / BEiT convention, decay 0.75).

    Supports HF VideoMAE (``encoder.layer.*``) and VideoMAE2 remote-code
    (``model.blocks.*`` / ``model.patch_embed``). Deeper layers get a larger LR;
    bias / 1-D (norm) params get no weight decay. Frozen params are skipped.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None and hasattr(model, "encoder"):
        backbone = model.encoder.backbone
    if backbone is None:
        raise AttributeError("Model has no backbone for layer-wise LR decay.")
    num_layers = _backbone_num_layers(backbone) + 1  # + head group

    def layer_id(name: str) -> int:
        for prefix in ("backbone.", "encoder.backbone."):
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix) :]
            if rest.startswith("embeddings") or rest.startswith("model.patch_embed"):
                return 0
            if "patch_embed" in rest.split(".")[:2]:
                return 0
            # HF VideoMAE
            marker = "encoder.layer."
            if marker in rest:
                return int(rest.split(marker)[1].split(".")[0]) + 1
            # VideoMAE2 remote code: model.blocks.{i}
            blocks_marker = "model.blocks."
            if blocks_marker in rest:
                return int(rest.split(blocks_marker)[1].split(".")[0]) + 1
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
    if not groups:
        raise RuntimeError("layerwise_param_groups produced no trainable parameter groups.")
    return list(groups.values())
