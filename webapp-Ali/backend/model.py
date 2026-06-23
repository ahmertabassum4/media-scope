import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IN_DIM, HIDDEN, DROPOUT, N_FACT, MLP_PATH, FACTUALITY


class MLP(nn.Module):
    """Identical to notebook cell 38; forward ignores edge_index (no message passing)."""
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, num_classes=N_FACT, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index=None):
        return self.fc(self.net(x))


_model = None


def load_model():
    global _model
    state = torch.load(MLP_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    in_features = state["net.0.weight"].shape[1]
    assert in_features == IN_DIM == 1198, f"mlp first Linear in_features {in_features} != {IN_DIM}"
    m = MLP()
    m.load_state_dict(state)
    m.eval()
    _model = m
    return m


def predict(x_np):
    """x_np: (1198,) float32 -> (label, {class: confidence})."""
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()
    idx = int(max(range(N_FACT), key=lambda i: probs[i]))
    confidences = {FACTUALITY[i]: probs[i] for i in range(N_FACT)}
    return FACTUALITY[idx], confidences
