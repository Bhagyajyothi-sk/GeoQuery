"""
GeoQueryAI — RemoteCLIP Encoder.

Wraps the open_clip ViT-B-32 model loaded with the RemoteCLIP checkpoint.
This module is intentionally kept separate so the AI/ML engineer can inspect
and modify the encoding logic independently of the retrieval pipeline.

════════════════════════════════════════════════════════════════════
  PLUG-IN POINT for the AI/ML engineer
════════════════════════════════════════════════════════════════════
  This module implements the RemoteCLIP text/image encoder.
  The checkpoint loading pattern mirrors test_remoteclip.py exactly.

  To swap the checkpoint or architecture, modify:
    - ARCHITECTURE constant
    - The _load() classmethod

  Do NOT modify semantic_search() in interface.py to change
  encoding behavior — modify this module instead.
════════════════════════════════════════════════════════════════════

Checkpoint verified:
  Path : models/RemoteCLIP-ViT-B-32.pt
  Keys : 0 missing, 0 unexpected
  Dim  : 512
  Norm : L2-normalized before similarity search
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("geoquery.remoteclip")

# ── Architecture constants ─────────────────────────────────────────────────────
ARCHITECTURE = "ViT-B-32"
EMBEDDING_DIM = 512

# Prompt templates for optional ensemble encoding.
# Averaging multiple prompts improves zero-shot retrieval quality.
_DEFAULT_PROMPT_TEMPLATES = [
    "{query}",
    "satellite image of {query}",
    "remote sensing image showing {query}",
    "aerial view of {query}",
]


class RemoteCLIPEncoder:
    """
    Singleton wrapper around the RemoteCLIP ViT-B-32 model.

    Usage:
        encoder = RemoteCLIPEncoder.instance()
        text_vec = encoder.encode_text(["large water body"])   # (1, 512)
        img_vec  = encoder.encode_image([pil_image])           # (1, 512)
    """

    _lock: threading.Lock = threading.Lock()
    _encoder: Optional["RemoteCLIPEncoder"] = None

    def __init__(self, model, preprocess, tokenizer, device: torch.device) -> None:
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._device = device

    # ── Singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls, checkpoint_path: Optional[str] = None) -> "RemoteCLIPEncoder":
        """
        Return the singleton encoder, loading the checkpoint on first call.

        Args:
            checkpoint_path: Absolute or project-relative path to the .pt file.
                             If None, resolved from settings.remoteclip_model_path.

        Raises:
            SearchServiceError: If the checkpoint file is missing or corrupt.
        """
        if cls._encoder is not None:
            return cls._encoder

        with cls._lock:
            if cls._encoder is not None:  # double-checked locking
                return cls._encoder
            cls._encoder = cls._load(checkpoint_path)
            return cls._encoder

    @classmethod
    def _load(cls, checkpoint_path: Optional[str]) -> "RemoteCLIPEncoder":
        """Load and verify the RemoteCLIP checkpoint. Called exactly once."""
        # Deferred import — avoids circular imports with config/errors
        from app.core.config import settings
        from app.core.errors import SearchServiceError

        try:
            import open_clip  # noqa: PLC0415
        except ImportError as exc:
            raise SearchServiceError(
                "open_clip_torch is not installed. "
                "Run: pip install open-clip-torch"
            ) from exc

        # ── Resolve checkpoint path ────────────────────────────────────────────
        raw_path = checkpoint_path or settings.remoteclip_model_path
        if not raw_path:
            raise SearchServiceError(
                "remoteclip_model_path is not configured. "
                "Set REMOTECLIP_MODEL_PATH in your .env file."
            )

        ckpt = Path(raw_path)
        if not ckpt.is_absolute():
            # Resolve relative to project root.
            # This file lives at: <project_root>/backend/app/services/search/remoteclip.py
            # parents: [0]=search, [1]=services, [2]=app, [3]=backend, [4]=project_root
            project_root = Path(__file__).resolve().parents[4]
            ckpt = project_root / raw_path

        if not ckpt.exists():
            raise SearchServiceError(
                f"RemoteCLIP checkpoint not found: {ckpt}. "
                "Verify REMOTECLIP_MODEL_PATH in your .env file."
            )

        # ── Build model architecture ───────────────────────────────────────────
        logger.info("RemoteCLIP: loading architecture %s ...", ARCHITECTURE)
        model, _, preprocess = open_clip.create_model_and_transforms(
            ARCHITECTURE,
            pretrained=None,  # load weights manually from checkpoint
        )
        tokenizer = open_clip.get_tokenizer(ARCHITECTURE)

        # ── Load checkpoint (same pattern as verified test_remoteclip.py) ──────
        logger.info("RemoteCLIP: loading checkpoint from %s ...", ckpt)
        state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(
                "RemoteCLIP: %d missing keys (first 5: %s)",
                len(missing),
                missing[:5],
            )
        if unexpected:
            logger.warning(
                "RemoteCLIP: %d unexpected keys (first 5: %s)",
                len(unexpected),
                unexpected[:5],
            )

        model.eval()

        # CPU-only: CUDA unavailable in this environment
        device = torch.device("cpu")
        model = model.to(device)

        logger.info(
            "RemoteCLIP: ready | architecture=%s | dim=%d | device=%s",
            ARCHITECTURE,
            EMBEDDING_DIM,
            device,
        )
        return cls(model, preprocess, tokenizer, device)

    @classmethod
    def reset(cls) -> None:
        """Release the singleton (used in tests to force reload)."""
        with cls._lock:
            cls._encoder = None

    # ── Public encoding API ────────────────────────────────────────────────────

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of text strings into normalized RemoteCLIP embeddings.

        Args:
            texts: List of query strings, e.g. ["large water body"].

        Returns:
            float32 numpy array of shape (len(texts), 512), L2-normalized.
        """
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        features = _l2_normalize(features)
        return features.cpu().float().numpy()

    def encode_image(self, images: list) -> np.ndarray:
        """
        Encode a list of PIL Images into normalized RemoteCLIP embeddings.

        Args:
            images: List of PIL.Image.Image objects (RGB, any size — preprocessed
                    automatically by the OpenCLIP transform pipeline).

        Returns:
            float32 numpy array of shape (len(images), 512), L2-normalized.
        """
        tensors = torch.stack(
            [self._preprocess(img) for img in images]
        ).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensors)
        features = _l2_normalize(features)
        return features.cpu().float().numpy()

    def encode_text_ensemble(
        self,
        query: str,
        templates: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Encode a query using multiple prompt templates, then average + renormalize.

        Improves zero-shot retrieval quality by averaging over multiple phrasings.

        Args:
            query:     Core visual query, e.g. "large water body".
            templates: f-string templates containing `{query}`.
                       Defaults to _DEFAULT_PROMPT_TEMPLATES.

        Returns:
            float32 numpy array of shape (1, 512), L2-normalized.
        """
        if templates is None:
            templates = _DEFAULT_PROMPT_TEMPLATES

        prompts = [t.format(query=query) for t in templates]
        embeddings = self.encode_text(prompts)              # (N_templates, 512)
        mean_vec = embeddings.mean(axis=0, keepdims=True)   # (1, 512)
        # Renormalize the averaged vector
        norm = np.linalg.norm(mean_vec, axis=-1, keepdims=True)
        return (mean_vec / (norm + 1e-8)).astype(np.float32)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize a tensor along its last dimension."""
    return tensor / tensor.norm(dim=-1, keepdim=True)
