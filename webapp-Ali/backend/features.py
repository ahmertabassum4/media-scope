import io
import re
import numpy as np
import torch
from PIL import Image

from config import (
    FACTUALITY, BIAS, GENRE, PROV_DIM, VERDICT_DIM, VIS_DIM, RAT_DIM, IN_DIM,
    PROV_KEYS, PROV_MU, PROV_SD, MINILM_ID, DINOV2_ID,
)

_MU = np.asarray(PROV_MU, dtype=np.float32)
_SD = np.asarray(PROV_SD, dtype=np.float32)


# ---- label normalisation (notebook cell 3) ----
def norm_label(s):
    if s is None:
        return ""
    return re.sub(r"[\s_]+", " ", str(s).upper().replace("-", " ").replace("_", " ")).strip()


_FACT_NORM = {norm_label(l): i for i, l in enumerate(FACTUALITY)}
_BIAS_NORM = {norm_label(l): i for i, l in enumerate(BIAS)}
_GENRE_NORM = {norm_label(l): i for i, l in enumerate(GENRE)}


# ---- lazy-loaded encoders ----
_dino = None
_dino_tf = None
_minilm = None


def init_encoders():
    global _dino, _dino_tf, _minilm
    import timm
    from sentence_transformers import SentenceTransformer
    _dino = timm.create_model(DINOV2_ID, pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({}, model=_dino)
    _dino_tf = timm.data.create_transform(**cfg)
    _minilm = SentenceTransformer(MINILM_ID, device="cpu")


# ---- visual embedding (notebook cell 24): raw DINOv2 output, not normalised ----
def visual_embedding(png_bytes):
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        x = _dino_tf(img).unsqueeze(0)
        with torch.no_grad():
            emb = _dino(x).squeeze(0).cpu().numpy()
        return emb.astype(np.float32)
    except Exception:
        return np.zeros(VIS_DIM, dtype=np.float32)


# ---- provenance vector (notebook cell 18) ----
def _flatten_numeric(prov, prefix=""):
    out = {}
    for k, v in prov.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_flatten_numeric(v, prefix=key + "."))
    return out


def provenance_vector(prov):
    """prov dict or None. None -> zero vector before standardisation (degraded row)."""
    fl = _flatten_numeric(prov) if prov else {}
    vec = np.zeros(PROV_DIM, dtype=np.float32)
    for j, key in enumerate(PROV_KEYS):
        vec[j] = fl.get(key, 0.0)
    return ((vec - _MU) / _SD).astype(np.float32)


# ---- verdict one-hot (notebook cell 20): [factuality 5 | bias 5 | genre 4] ----
def verdict_onehot(verdicts):
    vec = np.zeros(VERDICT_DIM, dtype=np.float32)
    fi = _FACT_NORM.get(norm_label(verdicts.get("factuality", "")))
    bi = _BIAS_NORM.get(norm_label(verdicts.get("bias", "")))
    gi = _GENRE_NORM.get(norm_label(verdicts.get("genre", "")))
    if fi is not None:
        vec[fi] = 1.0
    if bi is not None:
        vec[len(FACTUALITY) + bi] = 1.0
    if gi is not None:
        vec[len(FACTUALITY) + len(BIAS) + gi] = 1.0
    return vec


# ---- rationale embedding (notebook cell 22): MiniLM, normalised ----
def rationale_embedding(text):
    emb = _minilm.encode([text or ""], convert_to_numpy=True, normalize_embeddings=True)
    return emb[0].astype(np.float32)


# ---- assemble X (notebook cell 26) ----
def assemble(png_bytes, prov, verdicts, rationale):
    vis = visual_embedding(png_bytes)
    pv = provenance_vector(prov)
    vd = verdict_onehot(verdicts)
    rat = rationale_embedding(rationale)
    assert vis.shape[0] == VIS_DIM and pv.shape[0] == PROV_DIM
    assert vd.shape[0] == VERDICT_DIM and rat.shape[0] == RAT_DIM
    x = np.concatenate([vis, pv, vd, rat], axis=0).astype(np.float32)
    assert x.shape[0] == IN_DIM == 1198, f"assembled {x.shape[0]} != {IN_DIM}"
    return x
