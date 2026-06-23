"""Feature assembly for Ali's factuality MLP, ported from the training notebook.

Feature blocks (order must match training):

- ``vis``     : DINOv2 ViT-B/14 image embedding (VIS_DIM, raw, not normalised)
- ``prov``    : standardised provenance vector (PROV_DIM) — metadata model only
- ``verdict`` : joint-verdict one-hot [factuality | bias | genre] (VERDICT_DIM)
- ``rat``     : MiniLM rationale embedding (RAT_DIM, mean-pooled + L2 normalised)

Metadata vector = [vis, prov, verdict, rat]; image-only vector = [vis, verdict, rat].
The MiniLM embedding is computed with ``transformers`` (mean pooling + L2 norm), which
reproduces ``SentenceTransformer('all-MiniLM-L6-v2').encode(normalize_embeddings=True)``
used at training time, without adding a sentence-transformers dependency.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DINOV2_ID = "vit_base_patch14_dinov2.lvd142m"
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Canonical label orderings from the training notebook. Used as a fallback when the
# exported factuality_config.json omits these keys; the verdict one-hot layout
# [factuality | bias | genre] must match the order the MLP was trained on.
DEFAULT_FACTUALITY = ["VERY LOW", "LOW", "MIXED", "HIGH", "VERY HIGH"]
DEFAULT_BIAS = ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"]
DEFAULT_GENRE = ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"]


def norm_label(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[\s_]+", " ", str(value).upper().replace("-", " ").replace("_", " ")).strip()


class FactualityFeatures:
    def __init__(self, model_dir: Path, config: dict[str, Any]) -> None:
        stats = json.loads((model_dir / "factuality_prov_stats.json").read_text(encoding="utf-8"))
        self.prov_keys = list(stats["PROV_KEYS"])
        self.mu = np.asarray(stats["mu"], dtype=np.float32)
        self.sd = np.asarray(stats["sd"], dtype=np.float32)

        self.vis_dim = int(config["VIS_DIM"])
        self.prov_dim = int(config["PROV_DIM"])
        self.verdict_dim = int(config["VERDICT_DIM"])
        self.rat_dim = int(config["RAT_DIM"])
        self.factuality = list(config.get("FACTUALITY", DEFAULT_FACTUALITY))
        self.bias = list(config.get("BIAS", DEFAULT_BIAS))
        self.genre = list(config.get("GENRE", DEFAULT_GENRE))

        self._fact_norm = {norm_label(label): i for i, label in enumerate(self.factuality)}
        self._bias_norm = {norm_label(label): i for i, label in enumerate(self.bias)}
        self._genre_norm = {norm_label(label): i for i, label in enumerate(self.genre)}

        self._dino = None
        self._dino_tf = None
        self._minilm: tuple[Any, Any] | None = None
        self._lock = threading.Lock()

    # ---- lazy encoders (heavy; loaded on first use, shared across requests) ----
    def _ensure_dino(self) -> None:
        if self._dino is not None:
            return
        with self._lock:
            if self._dino is None:
                import timm

                model = timm.create_model(DINOV2_ID, pretrained=True, num_classes=0).eval()
                cfg = timm.data.resolve_data_config({}, model=model)
                self._dino_tf = timm.data.create_transform(**cfg)
                self._dino = model

    def _ensure_minilm(self) -> None:
        if self._minilm is not None:
            return
        with self._lock:
            if self._minilm is None:
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(MINILM_ID)
                model = AutoModel.from_pretrained(MINILM_ID).eval()
                self._minilm = (tokenizer, model)

    # ---- feature blocks ----
    def visual_embedding(self, image: Image.Image) -> np.ndarray:
        import torch

        self._ensure_dino()
        try:
            tensor = self._dino_tf(image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                emb = self._dino(tensor).squeeze(0).cpu().numpy()
            return emb.astype(np.float32)
        except Exception:
            return np.zeros(self.vis_dim, dtype=np.float32)

    def _flatten_numeric(self, prov: dict[str, Any], prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in prov.items():
            full = f"{prefix}{key}"
            if isinstance(value, bool):
                out[full] = float(value)
            elif isinstance(value, (int, float)):
                out[full] = float(value)
            elif isinstance(value, dict):
                out.update(self._flatten_numeric(value, full + "."))
        return out

    def provenance_vector(self, prov: dict[str, Any] | None) -> np.ndarray:
        flat = self._flatten_numeric(prov) if prov else {}
        vec = np.zeros(self.prov_dim, dtype=np.float32)
        for j, key in enumerate(self.prov_keys):
            vec[j] = flat.get(key, 0.0)
        return ((vec - self.mu) / self.sd).astype(np.float32)

    def verdict_onehot(self, verdicts: dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.verdict_dim, dtype=np.float32)
        fi = self._fact_norm.get(norm_label(verdicts.get("factuality", "")))
        bi = self._bias_norm.get(norm_label(verdicts.get("bias", "")))
        gi = self._genre_norm.get(norm_label(verdicts.get("genre", "")))
        if fi is not None:
            vec[fi] = 1.0
        if bi is not None:
            vec[len(self.factuality) + bi] = 1.0
        if gi is not None:
            vec[len(self.factuality) + len(self.bias) + gi] = 1.0
        return vec

    def rationale_embedding(self, text: str | None) -> np.ndarray:
        import torch

        self._ensure_minilm()
        tokenizer, model = self._minilm
        inputs = tokenizer([text or ""], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normed.squeeze(0).cpu().numpy().astype(np.float32)

    def assemble(
        self,
        image: Image.Image,
        verdicts: dict[str, Any],
        rationale: str,
        prov: dict[str, Any] | None,
    ) -> np.ndarray:
        vis = self.visual_embedding(image)
        verdict = self.verdict_onehot(verdicts)
        rat = self.rationale_embedding(rationale)
        if prov is not None:
            return np.concatenate([vis, self.provenance_vector(prov), verdict, rat]).astype(np.float32)
        return np.concatenate([vis, verdict, rat]).astype(np.float32)
