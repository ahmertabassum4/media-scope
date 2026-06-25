"""Ali's factuality MLP(s), replacing the previous ExtraTrees pipeline.

Two trained variants are loaded from the model directory:

- ``factuality_metadata_mlp.pt`` — full feature vector incl. URL-derived provenance
  (META_IN_DIM, e.g. 1198). Used when the analyze request came from a URL.
- ``factuality_image_mlp.pt`` — screenshot-only features, no provenance block
  (IMAGE_IN_DIM, e.g. 1166). Used for plain image uploads.

The architecture and feature contract mirror the training notebook; see
``factuality_config.json`` (labels, dims, hidden, dropout) produced alongside the
weights by the Colab export cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Identical to the notebook MLP; ``forward`` ignores edge_index (no message passing)."""

    def __init__(self, in_dim: int, hidden: int = 256, num_classes: int = 5, dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:  # noqa: ARG002
        return self.fc(self.net(x))


class FactualityModels:
    def __init__(self, model_dir: Path) -> None:
        config_path = model_dir / "factuality_config.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        # Labels are lowercased to match the bias labels and the frontend spectrum keys.
        self.labels = [str(label).lower() for label in self.config["FACTUALITY"]]
        self.n_fact = int(self.config["N_FACT"])
        self.hidden = int(self.config.get("hidden", 256))
        self.dropout = float(self.config.get("dropout", 0.4))
        self.meta_in_dim = int(self.config["META_IN_DIM"])
        self.image_in_dim = int(self.config["IMAGE_IN_DIM"])
        self.name = "ali_factuality_mlp"
        self.meta_model = self._load(model_dir / "factuality_metadata_mlp.pt", self.meta_in_dim)
        self.image_model = self._load(model_dir / "factuality_image_mlp.pt", self.image_in_dim)

    def _load(self, path: Path, in_dim: int) -> MLP:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        got = state["net.0.weight"].shape[1]
        if got != in_dim:
            raise ValueError(f"{path.name}: first Linear in_features {got} != configured in_dim {in_dim}")
        model = MLP(in_dim, hidden=self.hidden, num_classes=self.n_fact, dropout=self.dropout)
        model.load_state_dict(state)
        model.eval()
        return model

    def predict(self, x: np.ndarray, use_metadata: bool) -> tuple[str, dict[str, float]]:
        model = self.meta_model if use_metadata else self.image_model
        tensor = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze(0).tolist()
        best = int(max(range(self.n_fact), key=lambda i: probs[i]))
        confidences = {self.labels[i]: float(probs[i]) for i in range(self.n_fact)}
        return self.labels[best], confidences
