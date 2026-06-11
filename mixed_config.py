# mixed_config.py
# Single source of truth for the MIXED (5-class factuality) experiments, so the
# sampler, inference scripts, and eval scripts all agree on labels and paths.

from pathlib import Path

# Five-class factuality scale (MIXED reinstated between LOW and HIGH).
FACT_LABELS = ["VERY LOW", "LOW", "MIXED", "HIGH", "VERY HIGH"]

# Bias / genre label spaces are unchanged; re-exported here for convenience.
BIAS_LABELS  = ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"]
GENRE_LABELS = ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"]

# Sampled subset produced by sample_mixed_subset.py
SAMPLE_INDEX     = "snapshot_sample_mixed_index.csv"
SAMPLE_IMAGE_DIR = "sample_snapshots_mixed"

# Per-task inference output folders (kept separate and tidy).
RESULTS_DIRS = {
    "factuality": Path("mixed_results_factuality"),
    "bias":       Path("mixed_results_bias"),
    "genre":      Path("mixed_results_genre"),
    "joint":      Path("mixed_results_joint"),
}

# Eval plots: one master folder, one sub-folder per task.
PLOTS_MASTER = "eval_plots_mixed"
PLOTS_DIRS = {
    "factuality": f"{PLOTS_MASTER}/eval_plots_factuality",
    "bias":       f"{PLOTS_MASTER}/eval_plots_bias",
    "genre":      f"{PLOTS_MASTER}/eval_plots_genre",
    "joint":      f"{PLOTS_MASTER}/eval_plots_joint",
}

# Column in the source index that overrides `factuality` for Mixed_output rows.
MIXED_TIER_COLUMN = "Mixed Factuality"


def effective_tier(row):
    """Return the row's true 5-class factuality tier.

    Prefer the new 'Mixed Factuality' column (populated for the Mixed_output
    captures); otherwise fall back to the standard 'factuality' column. Returns
    the upper-cased tier if it is one of FACT_LABELS, else None.
    """
    mixed = (row.get(MIXED_TIER_COLUMN) or "").strip().upper()
    base  = (row.get("factuality") or "").strip().upper()
    tier = mixed or base
    return tier if tier in FACT_LABELS else None