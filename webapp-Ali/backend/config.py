import json
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "mediascope_artifacts"

with open(ARTIFACTS_DIR / "config.json", encoding="utf-8") as f:
    CFG = json.load(f)

with open(ARTIFACTS_DIR / "prov_stats.json", encoding="utf-8") as f:
    PROV_STATS = json.load(f)

FACTUALITY = CFG["FACTUALITY"]
BIAS = CFG["BIAS"]
GENRE = CFG["GENRE"]
N_FACT = CFG["N_FACT"]
PROV_DIM = CFG["PROV_DIM"]
VERDICT_DIM = CFG["VERDICT_DIM"]
VIS_DIM = CFG["VIS_DIM"]
RAT_DIM = CFG["RAT_DIM"]
IN_DIM = CFG["IN_DIM"]
FEATURE_ORDER = CFG["feature_order"]
HIDDEN = CFG["hidden"]
DROPOUT = CFG["dropout"]

PROV_KEYS = PROV_STATS["PROV_KEYS"]
PROV_MU = PROV_STATS["mu"]
PROV_SD = PROV_STATS["sd"]

MLP_PATH = ARTIFACTS_DIR / "mlp.pt"

# Joint-verdict model + prompt are taken verbatim from the training pipeline
# (gpt5_5_inference.py -> inference_joint.py) so serving features match training.
JOINT_MODEL_ID = "openai/gpt-5.5"
EXPLAIN_MODEL_ID = "openai/gpt-5.5"

MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
DINOV2_ID = "vit_base_patch14_dinov2.lvd142m"
