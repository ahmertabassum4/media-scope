"""Vision-only graph factuality model, faithful to mediascope_graph_structure.ipynb.

Pipeline (the deployable `visual@0.80 / prop / xgboost` variant):

  struct_vis  = standardise(DINOv2)                         # 768, using train vis_mu/vis_sd
  graph       = visual@0.80 (cosine on L2-normalised DINOv2)
  prop        = 2-step symmetric-normalised neighbour diffusion of struct_vis
  features    = [ prop(struct_vis) | verdict one-hot | MiniLM rationale ] = 1166
  classifier  = XGBoost

The 555 training nodes are shipped as a frozen corpus. At inference a new screenshot is
attached to that corpus by cosine >= threshold, the diffusion is run over corpus+node, and
the new node's propagated vector feeds the classifier. No provenance is used, so the same
path serves URL captures and image uploads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


def _symmetric_cosine_edges(visnorm: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edges (returned symmetric) between rows with cosine >= threshold.

    Rows of ``visnorm`` are unit-norm, so ``visnorm @ visnorm.T`` is the cosine matrix.
    """
    sim = visnorm @ visnorm.T
    np.fill_diagonal(sim, 0.0)
    iu, ju = np.where(np.triu(sim >= threshold, k=1))
    src = np.concatenate([iu, ju]).astype(np.int64)
    dst = np.concatenate([ju, iu]).astype(np.int64)
    return src, dst


class GraphVisionFactuality:
    def __init__(self, model_dir: Path) -> None:
        config = json.loads((model_dir / "factuality_config.json").read_text(encoding="utf-8"))
        stats = json.loads((model_dir / "factuality_stats.json").read_text(encoding="utf-8"))

        self.labels = [str(label).lower() for label in config["FACTUALITY"]]
        self.n_fact = int(config["N_FACT"])
        self.vis_dim = int(config["VIS_DIM"])
        self.verdict_dim = int(config["VERDICT_DIM"])
        self.rat_dim = int(config["RAT_DIM"])
        self.feat_dim = int(config["FEAT_DIM"])
        self.threshold = float(config["graph_threshold"])
        self.steps = int(config["prop_steps"])
        self.name = "graph_vision_xgb"

        self.vis_mu = np.asarray(stats["vis_mu"], dtype=np.float32)
        self.vis_sd = np.asarray(stats["vis_sd"], dtype=np.float32)

        # Frozen training corpus: standardised DINOv2 (for diffusion) and L2-normalised
        # DINOv2 (for building visual edges to a new node).
        self.corpus_struct = np.load(model_dir / "factuality_train_struct_vis.npy").astype(np.float32)
        self.corpus_visnorm = np.load(model_dir / "factuality_train_visnorm.npy").astype(np.float32)
        self.n_nodes = self.corpus_struct.shape[0]

        # Corpus-corpus edges are fixed; precompute once so each request only adds the new node.
        self.cc_src, self.cc_dst = _symmetric_cosine_edges(self.corpus_visnorm, self.threshold)

        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "factuality_xgb.json"))

        # High-reliability blend: when the vision-LLM verdict is HIGH / VERY HIGH, mix its
        # one-hot back into the XGBoost probabilities. The model under-predicts the top
        # tiers (VERY HIGH recall ~0); 5-fold CV showed alpha=0.3 on high-tier verdicts
        # lifts macro-F1 0.60 -> 0.67 (VERY HIGH recall 0.00 -> 0.27) with no accuracy loss.
        # alpha > ~0.3 starts collapsing MIXED, so keep it modest. Tunable without a rebuild.
        self.blend_alpha = float(os.environ.get("FACTUALITY_LLM_BLEND_ALPHA", "0.3"))
        self.high_tiers = {i for i, lab in enumerate(self.labels) if lab in ("high", "very high")}

    def _propagate_new(self, struct_new: np.ndarray, visnorm_new: np.ndarray) -> np.ndarray:
        """Return the new node's 2-step diffused structural vector over corpus + new node."""
        new_idx = self.n_nodes
        total = self.n_nodes + 1
        x = np.vstack([self.corpus_struct, struct_new[None, :]]).astype(np.float32)

        # new node <-> corpus nodes with cosine >= threshold
        sims = self.corpus_visnorm @ visnorm_new
        nb = np.where(sims >= self.threshold)[0]
        if nb.size:
            src = np.concatenate([self.cc_src, nb, np.full(nb.size, new_idx, dtype=np.int64)])
            dst = np.concatenate([self.cc_dst, np.full(nb.size, new_idx, dtype=np.int64), nb])
        else:
            # isolated new node: only the corpus graph is present; its diffused value is
            # just its own features (matches the notebook's empty-graph identity case).
            src, dst = self.cc_src, self.cc_dst

        if src.size == 0:
            return struct_new

        deg = np.bincount(np.concatenate([src, dst]), minlength=total).astype(np.float32)
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
        w = (dinv[src] * dinv[dst]).astype(np.float32)
        h = x.copy()
        for _ in range(self.steps):
            agg = np.zeros_like(h)
            np.add.at(agg, dst, w[:, None] * h[src])
            h = 0.5 * h + 0.5 * agg
        return h[new_idx]

    def predict(
        self,
        raw_vis: np.ndarray,
        verdict_onehot: np.ndarray,
        rationale_emb: np.ndarray,
    ) -> tuple[str, dict[str, float]]:
        raw_vis = np.asarray(raw_vis, dtype=np.float32).ravel()
        struct_new = (raw_vis - self.vis_mu) / self.vis_sd
        norm = float(np.linalg.norm(raw_vis))
        visnorm_new = raw_vis / norm if norm > 0 else raw_vis

        prop = self._propagate_new(struct_new.astype(np.float32), visnorm_new.astype(np.float32))
        feat = np.concatenate([prop, np.asarray(verdict_onehot, np.float32).ravel(),
                               np.asarray(rationale_emb, np.float32).ravel()]).astype(np.float32)
        if feat.shape[0] != self.feat_dim:
            raise ValueError(f"feature dim {feat.shape[0]} != expected {self.feat_dim}")

        proba = self.booster.predict(xgb.DMatrix(feat.reshape(1, -1)))[0].astype(np.float64)

        # S2 high-tier-only blend (see __init__): only adjust when the LLM is confident the
        # source is reliable. verdict_onehot[:n_fact] is the LLM factuality one-hot (it never
        # marks MIXED), so its argmax is the LLM tier; sum==0 means no/unknown verdict.
        fact_oh = np.asarray(verdict_onehot, dtype=np.float64).ravel()[: self.n_fact]
        if self.blend_alpha > 0 and fact_oh.sum() > 0:
            llm_tier = int(np.argmax(fact_oh))
            if llm_tier in self.high_tiers:
                proba = (1.0 - self.blend_alpha) * proba + self.blend_alpha * fact_oh

        best = int(np.argmax(proba))
        confidences = {self.labels[i]: float(proba[i]) for i in range(self.n_fact)}
        return self.labels[best], confidences
