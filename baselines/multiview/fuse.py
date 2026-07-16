"""Multi-view fusion + benchmark table (paper §4.2.3, §5).

Implements every fusion strategy from the paper over the five views:
  svm        - concat + linear-kernel SVM, max_iter=60, tol=0.01 (their spec,
               "following Baly et al." -> MinMaxScaler before the SVM)
  mlp        - single-hidden-layer perceptron on the concatenation
  self_attn  - each view projected to a 64-d token, one self-attention layer
               over the view tokens, mean-pool, linear head
  cross_attn - text tokens attend over graph tokens (one direction)
  co_attn    - both directions, pooled outputs concatenated
  rl         - PPO contextual bandit (stable-baselines3): state = concat of
               per-view 64-d projections, action w in [0,1]^k,
               E_fused = sum_k w_k v_k, reward = P(y_true | E_fused) under a
               frozen classifier; gamma=0, policy MLP 2x128 tanh, lr 1e-4,
               batch 256, rollout 1024 (paper §4.2 + §5.1)

Protocol: fusion models are fit on TRAIN outlets; the torch fusions carve a
stratified 10% dev slice out of the existing train split for model selection;
the fixed TEST split is evaluated exactly once per row.

View spec strings: alexa_gcn / alexa_sage / alexa_resgated, hyperlink_*,
llm_* (graph views, 64-d) and articles / wiki (text views, 768-d, per task).
By default, missing or invalid embedding archives stop the run before any final
output is written. ``--allow-partial`` writes a clearly marked partial report.

Output: results/current/multiview_<task>.csv with the standard metric columns
(accuracy, balanced/macro accuracy, macro P/R/F1, per-class F1, MAE, MSE).

Run:  /opt/anaconda3/bin/python3 baselines/multiview/fuse.py --task factuality
"""

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    ARTICLES_JSONL,
    EMB_DIM_GNN,
    GRAPH_VIEWS,
    GRAPHS_DIR,
    RESULTS_DIR,
    ROOT,
    TASKS,
    atomic_write_text,
    build_content_filter,
    emb_path,
    exclusive_file_lock,
    load_embeddings,
    load_outlets,
    match_path,
    outlet_universe_sha256,
    outlet_label,
    scores,
    sha256_file,
    task_split,
)

SEED = 42
TOKEN_DIM = EMB_DIM_GNN  # views are projected to 64-d tokens for attention/RL
DEV_FRAC = 0.10          # carved from this project's existing training split
TORCH_EPOCHS, TORCH_PATIENCE, TORCH_LR, TORCH_BATCH = 200, 20, 1e-3, 64
RL_TIMESTEPS = 100_000
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
WIKI_PAGES = ROOT / "data" / "wiki" / "wiki_pages.jsonl"
_SOURCE_HASH_CACHE = {}
_GRAPH_FINGERPRINT_CACHE = {}
_ARTICLE_FILTER_SHA256 = None

# All graph encoders are useful diagnostics; Tables 7/8 report a subset.
GRAPH_ABLATION_ROWS = [
    ("majority", []),
    ("svm", ["alexa_gcn"]),
    ("svm", ["alexa_sage"]),
    ("svm", ["alexa_resgated"]),
    ("svm", ["hyperlink_gcn"]),
    ("svm", ["hyperlink_sage"]),
    ("svm", ["hyperlink_resgated"]),
    ("svm", ["llm_gcn"]),
    ("svm", ["llm_sage"]),
    ("svm", ["llm_resgated"]),
]

# MBFC-2025 combinations from Tables 5/6, adapted to the current label split.
TASK_FUSION_ROWS = {
    "bias": [
        ("svm", ["articles", "wiki"]),
        ("svm", ["hyperlink_gcn", "articles", "wiki"]),
        ("mlp", ["llm_gcn", "hyperlink_sage", "articles", "wiki"]),
        ("self_attn", ["llm_sage", "llm_gcn", "hyperlink_sage", "articles", "wiki"]),
        ("rl", ["alexa_gcn", "hyperlink_sage", "llm_gcn", "articles", "wiki"]),
    ],
    "factuality": [
        ("cross_attn", ["articles", "wiki"]),
        ("co_attn", ["alexa_resgated", "articles", "wiki"]),
        ("cross_attn", ["llm_gcn", "hyperlink_sage", "articles", "wiki"]),
        ("cross_attn", ["llm_sage", "llm_gcn", "hyperlink_gcn", "articles", "wiki"]),
        ("rl", ["alexa_gcn", "hyperlink_sage", "llm_gcn", "articles", "wiki"]),
    ],
}


def benchmark_rows(task):
    return GRAPH_ABLATION_ROWS + TASK_FUSION_ROWS[task]


def current_source_sha256(path):
    path = Path(path)
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache_key not in _SOURCE_HASH_CACHE:
        _SOURCE_HASH_CACHE[cache_key] = sha256_file(path)
    return _SOURCE_HASH_CACHE[cache_key]


def current_graph_fingerprint(graph):
    if graph not in _GRAPH_FINGERPRINT_CACHE:
        from gnn_embed import (graph_source_fingerprint,
                               validate_llm_graph_manifest,
                               validate_pinned_graph_sources)
        if graph == "llm":
            validate_llm_graph_manifest(load_outlets())
        else:
            validate_pinned_graph_sources(graph)
        _GRAPH_FINGERPRINT_CACHE[graph] = graph_source_fingerprint(
            graph, match_path(graph),
        )
    return _GRAPH_FINGERPRINT_CACHE[graph]


def current_article_filter_sha256():
    global _ARTICLE_FILTER_SHA256
    if _ARTICLE_FILTER_SHA256 is None:
        _ARTICLE_FILTER_SHA256 = build_content_filter(load_outlets())["fingerprint"]
    return _ARTICLE_FILTER_SHA256


def view_archive(spec, task):
    """Path of the embedding archive for one view spec string."""
    if spec in ("articles", "wiki"):
        return emb_path(spec, task=task)
    graph, _, encoder = spec.partition("_")
    if graph in GRAPH_VIEWS and encoder in ("gcn", "sage", "resgated"):
        return emb_path(graph, encoder=encoder)
    raise ValueError(f"unknown view spec {spec!r}")


def load_views(specs, task, keys, universe_sha256=None):
    """{spec: (n_outlets, dim) matrix aligned to keys}; raises if missing."""
    views = {}
    if universe_sha256 is None:
        universe_sha256 = outlet_universe_sha256(load_outlets())
    for spec in specs:
        path = view_archive(spec, task)
        if not path.exists():
            raise FileNotFoundError(path)
        expected = {
            "schema": "multiview_embedding_v2",
            "complete": True,
            "outlet_universe_sha256": universe_sha256,
        }
        if spec in ("articles", "wiki"):
            expected.update({
                "view": spec,
                "task": task,
                "model": TEXT_MODEL_NAME,
                "model_revision": TEXT_MODEL_REVISION,
                "epochs": 6,
                "max_len": 256,
                "source_sha256": current_source_sha256(
                    WIKI_PAGES if spec == "wiki" else ARTICLES_JSONL
                ),
                "content_filter_sha256": (
                    "" if spec == "wiki" else current_article_filter_sha256()
                ),
            })
        else:
            graph, _, encoder = spec.partition("_")
            expected.update({
                "graph": graph,
                "encoder": encoder,
                "epochs": 50,
                "contrast": ("dgi_feature_shuffle_v1" if graph == "alexa"
                             else "dgi_edge_destination_permutation_v1"),
                "match_sha256": current_source_sha256(match_path(graph)),
                "graph_source_sha256": current_graph_fingerprint(graph),
            })
        _, X = load_embeddings(
            path, expected_keys=keys, expected_metadata=expected,
        )
        expected_dim = 768 if spec in ("articles", "wiki") else EMB_DIM_GNN
        if X.shape[1] != expected_dim:
            raise ValueError(
                f"{path}: embedding width is {X.shape[1]}, expected {expected_dim}"
            )
        views[spec] = X
    return views


def dev_carve(train_keys, labels_by_key, seed=SEED):
    from sklearn.model_selection import train_test_split
    y = [labels_by_key[k] for k in train_keys]
    fit_keys, dev_keys = train_test_split(
        train_keys, test_size=DEV_FRAC, stratify=y, random_state=seed)
    return sorted(fit_keys), sorted(dev_keys)


def rows_for(keys, all_keys, X):
    index = {k: i for i, k in enumerate(all_keys)}
    return X[[index[k] for k in keys]]


# --------------------------------------------------------------------------
# fusion models
# --------------------------------------------------------------------------

def fit_predict_svm(train_matrix, train_labels, test_matrix):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    import warnings
    scaler = MinMaxScaler().fit(train_matrix)
    classifier = SVC(kernel="linear", max_iter=60, tol=0.01, random_state=SEED)
    with warnings.catch_warnings():
        import sklearn.exceptions
        warnings.simplefilter("ignore", sklearn.exceptions.ConvergenceWarning)
        classifier.fit(scaler.transform(train_matrix), train_labels)
    if classifier.fit_status_ != 0:
        print("    warning: paper-spec SVM reached max_iter=60 before convergence", flush=True)
    return classifier.predict(scaler.transform(test_matrix))


class ViewTokens(nn.Module):
    """Per-view linear projection to a shared 64-d token."""

    def __init__(self, dims, token_dim=TOKEN_DIM):
        super().__init__()
        self.projs = nn.ModuleList(nn.Linear(d, token_dim, bias=False) for d in dims)

    def forward(self, views):  # list of (B, d_k) -> (B, K, token_dim)
        return torch.stack([p(v) for p, v in zip(self.projs, views)], dim=1)


class MLPFusion(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(hidden, n_classes))

    def forward(self, views):
        return self.net(torch.cat(views, dim=1))


class SelfAttnFusion(nn.Module):
    def __init__(self, dims, n_classes, token_dim=TOKEN_DIM, heads=4):
        super().__init__()
        self.tokens = ViewTokens(dims, token_dim)
        self.attn = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.head = nn.Linear(token_dim, n_classes)

    def forward(self, views):
        tokens = self.tokens(views)
        attended, _ = self.attn(tokens, tokens, tokens)
        return self.head(attended.mean(dim=1))


class CrossAttnFusion(nn.Module):
    """Query group attends over the other group (co=True adds the reverse)."""

    def __init__(self, dims, query_idx, n_classes, token_dim=TOKEN_DIM,
                 heads=4, co=False):
        super().__init__()
        self.query_idx = list(query_idx)
        self.other_idx = [i for i in range(len(dims)) if i not in self.query_idx]
        if not self.query_idx or not self.other_idx:
            raise ValueError("cross-attention needs two non-empty view groups")
        self.co = co
        self.tokens = ViewTokens(dims, token_dim)
        self.query_to_context = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.context_to_query = (
            nn.MultiheadAttention(token_dim, heads, batch_first=True) if co else None
        )
        self.head = nn.Linear(token_dim * (2 if co else 1), n_classes)

    def forward(self, views):
        tokens = self.tokens(views)
        query = tokens[:, self.query_idx]
        context = tokens[:, self.other_idx]
        pooled = [self.query_to_context(query, context, context)[0].mean(dim=1)]
        if self.co:
            pooled.append(self.context_to_query(context, query, query)[0].mean(dim=1))
        return self.head(torch.cat(pooled, dim=1))


def standardize(fit, *others):
    missing = [np.all(matrix == 0, axis=1) for matrix in (fit,) + others]
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True) + 1e-8
    scaled = [(matrix - mean) / std for matrix in (fit,) + others]
    for matrix, mask in zip(scaled, missing):
        matrix[mask] = 0.0
    return scaled


def train_torch_fusion(model, views_fit, y_fit, views_dev, y_dev,
                       epochs=TORCH_EPOCHS, patience=TORCH_PATIENCE):
    from sklearn.metrics import f1_score
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=TORCH_LR)
    loss_fn = nn.CrossEntropyLoss()
    fit_tensors = [torch.tensor(view, dtype=torch.float32) for view in views_fit]
    dev_tensors = [torch.tensor(view, dtype=torch.float32) for view in views_dev]
    fit_labels = torch.tensor(y_fit, dtype=torch.long)
    n_fit = fit_labels.shape[0]
    best_f1, best_state, stale_epochs = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n_fit)
        for start in range(0, n_fit, TORCH_BATCH):
            batch_idx = order[start:start + TORCH_BATCH]
            optimizer.zero_grad()
            logits = model([tensor[batch_idx] for tensor in fit_tensors])
            loss = loss_fn(logits, fit_labels[batch_idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_dev = model(dev_tensors).argmax(dim=1).numpy()
        f1 = f1_score(y_dev, pred_dev, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, stale_epochs = f1, 0
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


@torch.no_grad()
def predict_torch_fusion(model, views):
    model.eval()
    tensors = [torch.tensor(view, dtype=torch.float32) for view in views]
    return model(tensors).argmax(dim=1).numpy()


# --------------------------------------------------------------------------
# RL-PPO dynamic fusion (paper §4.2 "RL-based Dynamic Fusion" + §5.1 RL Agent)
# --------------------------------------------------------------------------

class FrozenFusedClassifier(nn.Module):
    """Per-view 64-d projections + classifier over the weighted-sum embedding.

    Pre-trained on train with equal weights, then frozen: the PPO reward is
    P(y_true | sum_k w_k v_k) under this classifier.
    """

    def __init__(self, dims, n_classes, token_dim=TOKEN_DIM):
        super().__init__()
        self.tokens = ViewTokens(dims, token_dim)
        self.head = nn.Sequential(nn.Linear(token_dim, 128), nn.ReLU(),
                                  nn.Linear(128, n_classes))

    def forward(self, views, weights=None):  # weights: (B, K)
        t = self.tokens(views)
        if weights is None:
            fused = t.mean(dim=1)
        else:
            fused = (t * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(fused)

    @torch.no_grad()
    def token_states(self, views):  # (B, K, token_dim) for the RL observation
        return self.tokens(views)


def make_bandit_env(states, y, classifier, n_views, rng_seed=SEED):
    import gymnasium as gym
    from gymnasium import spaces

    class FusionBanditEnv(gym.Env):
        """One-step contextual bandit: pick view weights for one outlet."""

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                (n_views * TOKEN_DIM,), np.float32)
            self.action_space = spaces.Box(0.0, 1.0, (n_views,), np.float32)
            self.rng = np.random.default_rng(rng_seed)
            self.i = 0

        def _obs(self):
            return states[self.i].reshape(-1).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.i = int(self.rng.integers(len(y)))
            return self._obs(), {}

        def step(self, action):
            w = torch.tensor(np.clip(action, 0.0, 1.0), dtype=torch.float32)
            tokens = torch.tensor(states[self.i], dtype=torch.float32)
            fused = (tokens * w.unsqueeze(-1)).sum(dim=0, keepdim=True)
            with torch.no_grad():
                probs = torch.softmax(classifier.head(fused), dim=-1)[0]
            reward = float(probs[y[self.i]])
            return self._obs(), reward, True, False, {}

    return FusionBanditEnv()


def fit_predict_rl(views_fit, y_fit, views_dev, y_dev, views_test, n_classes,
                   timesteps=RL_TIMESTEPS):
    from stable_baselines3 import PPO

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dims = [v.shape[1] for v in views_fit]
    classifier = FrozenFusedClassifier(dims, n_classes)
    classifier, dev_f1 = train_torch_fusion(  # equal-weight pre-training
        classifier, views_fit, y_fit, views_dev, y_dev)
    for p in classifier.parameters():
        p.requires_grad_(False)
    classifier.eval()
    print(f"    frozen classifier dev macro-F1 (equal weights): {dev_f1:.4f}", flush=True)

    with torch.no_grad():
        states = classifier.token_states(
            [torch.tensor(v, dtype=torch.float32) for v in views_fit]).numpy()
    env = make_bandit_env(states, np.asarray(y_fit), classifier, len(dims))
    agent = PPO(
        "MlpPolicy", env,
        gamma=0.0,                       # contextual bandit (paper)
        learning_rate=1e-4, batch_size=256, n_steps=1024,
        policy_kwargs={"net_arch": [128, 128], "activation_fn": torch.nn.Tanh},
        seed=SEED, verbose=0,
    )
    agent.learn(total_timesteps=timesteps, progress_bar=False)

    @torch.no_grad()
    def predict(views):
        tokens = classifier.token_states(
            [torch.tensor(v, dtype=torch.float32) for v in views])
        obs = tokens.reshape(tokens.shape[0], -1).numpy().astype(np.float32)
        w, _ = agent.predict(obs, deterministic=True)
        w = torch.tensor(np.clip(w, 0.0, 1.0), dtype=torch.float32)
        fused = (tokens * w.unsqueeze(-1)).sum(dim=1)
        return classifier.head(fused).argmax(dim=1).numpy()

    return predict(views_dev), predict(views_test)


# --------------------------------------------------------------------------
# benchmark driver
# --------------------------------------------------------------------------

def text_first_query_idx(specs):
    """Indices of text views: the attention query group (assumption noted)."""
    idx = [i for i, s in enumerate(specs) if s in ("articles", "wiki")]
    if not idx or len(idx) == len(specs):
        return [0]
    return idx


def run_row(method, specs, task, outlets, train_keys, test_keys, labels_by_key,
            rl_timesteps, return_predictions=False):
    classes = TASKS[task]["classes"]
    label2id = {c: i for i, c in enumerate(classes)}
    y_train = [labels_by_key[k] for k in train_keys]
    y_test = [labels_by_key[k] for k in test_keys]

    def finish(row, predictions):
        predictions = list(predictions)
        return (row, predictions) if return_predictions else row

    if method == "majority":
        top = Counter(y_train).most_common(1)[0][0]
        predictions = [top] * len(y_test)
        return finish(
            scores("majority", "Majority class", "-", y_test, predictions, task),
            predictions,
        )

    keys_all = sorted(outlets)
    views = load_views(specs, task, keys_all, outlet_universe_sha256(outlets))
    components = "+".join(specs)

    if method == "svm":
        train_matrix = np.concatenate(
            [rows_for(train_keys, keys_all, views[spec]) for spec in specs], axis=1)
        test_matrix = np.concatenate(
            [rows_for(test_keys, keys_all, views[spec]) for spec in specs], axis=1)
        y_pred = fit_predict_svm(train_matrix, y_train, test_matrix)
        return finish(
            scores("SVM", f"SVM: {components}", components, y_test, list(y_pred), task),
            y_pred,
        )

    # torch fusions: stratified dev carve out of train, z-scored inputs
    fit_keys, dev_keys = dev_carve(train_keys, labels_by_key)
    fit_labels = [label2id[labels_by_key[k]] for k in fit_keys]
    dev_labels = [label2id[labels_by_key[k]] for k in dev_keys]
    fit_views, dev_views, test_views = [], [], []
    for spec in specs:
        fit_scaled, dev_scaled, test_scaled = standardize(
            rows_for(fit_keys, keys_all, views[spec]),
            rows_for(dev_keys, keys_all, views[spec]),
            rows_for(test_keys, keys_all, views[spec]))
        fit_views.append(fit_scaled)
        dev_views.append(dev_scaled)
        test_views.append(test_scaled)
    dims = [view.shape[1] for view in fit_views]

    if method == "rl":
        _, pred_ids = fit_predict_rl(fit_views, fit_labels, dev_views, dev_labels,
                                     test_views, len(classes),
                                     timesteps=rl_timesteps)
        y_pred = [classes[i] for i in pred_ids]
        return finish(
            scores("RL (PPO)", f"RL (PPO): {components}", components,
                   y_test, y_pred, task),
            y_pred,
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if method == "mlp":
        model = MLPFusion(sum(dims), len(classes))
    elif method == "self_attn":
        model = SelfAttnFusion(dims, len(classes))
    elif method == "cross_attn":
        model = CrossAttnFusion(dims, text_first_query_idx(specs), len(classes), co=False)
    elif method == "co_attn":
        model = CrossAttnFusion(dims, text_first_query_idx(specs), len(classes), co=True)
    else:
        raise ValueError(f"unknown method {method!r}")
    model, dev_f1 = train_torch_fusion(model, fit_views, fit_labels,
                                       dev_views, dev_labels)
    print(f"    dev macro-F1: {dev_f1:.4f}", flush=True)
    pred_ids = predict_torch_fusion(model, test_views)
    y_pred = [classes[i] for i in pred_ids]
    name = {"mlp": "MLP", "self_attn": "Self-attention",
            "cross_attn": "Cross-attention", "co_attn": "Co-attention"}[method]
    return finish(
        scores(name, f"{name}: {components}", components, y_test, y_pred, task),
        y_pred,
    )


def run_benchmark(args):
    only = {m.strip() for m in args.only.split(",") if m.strip()}
    configured_rows = benchmark_rows(args.task)
    valid_methods = {method for method, _ in configured_rows}
    unknown_methods = sorted(only - valid_methods)
    if unknown_methods:
        raise SystemExit(f"unknown --only methods: {unknown_methods}")

    outlets = load_outlets()
    train_keys, test_keys = task_split(outlets, args.task)
    labels_by_key = {k: outlet_label(outlets[k], args.task)
                     for k in train_keys + test_keys}
    print(f"task={args.task}  train={len(train_keys)}  test={len(test_keys)}")

    requested = [(method, specs) for method, specs in configured_rows
                 if not only or method in only]
    if any(method == "rl" for method, _ in requested) and args.rl_timesteps < 1:
        raise SystemExit("--rl-timesteps must be at least 1 when RL is requested")
    skipped = []
    for method, specs in requested:
        missing = [spec for spec in specs if not view_archive(spec, args.task).exists()]
        if missing:
            skipped.append({"method": method, "specs": specs, "missing": missing})
    if skipped and not args.allow_partial:
        details = "; ".join(
            f"{item['method']}:{'+'.join(item['specs'])} missing {item['missing']}"
            for item in skipped
        )
        raise SystemExit(
            "required embedding archives are missing; generate them first or use "
            f"--allow-partial for a non-final report: {details}"
        )

    rows, prediction_rows = [], []
    for method, specs in requested:
        if only and method not in only:
            continue
        label = f"{method}: {'+'.join(specs) or '-'}"
        missing = [spec for spec in specs if not view_archive(spec, args.task).exists()]
        if missing:
            print(f"skip {label}: missing embeddings for {missing}")
            continue
        print(f"row: {label}", flush=True)
        try:
            row, predictions = run_row(
                method, specs, args.task, outlets, train_keys, test_keys,
                labels_by_key, args.rl_timesteps, return_predictions=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"invalid inputs for {label}: {exc}") from None
        rows.append(row)
        prediction_rows.extend({
            "experiment": row["experiment"],
            "key": key,
            "truth": labels_by_key[key],
            "pred": prediction,
        } for key, prediction in zip(test_keys, predictions))

    if not rows:
        raise SystemExit("no rows produced — generate embeddings first")
    complete = (
        not only and not skipped and len(rows) == len(configured_rows)
        and args.rl_timesteps == RL_TIMESTEPS
    )
    default_name = (f"multiview_{args.task}.csv" if complete
                    else f"multiview_{args.task}_partial.csv")
    out = args.out or RESULTS_DIR / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(metrics_buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(out, metrics_buffer.getvalue())
    predictions_out = out.with_name(f"{out.stem}_predictions.csv")
    predictions_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        predictions_buffer, fieldnames=["experiment", "key", "truth", "pred"],
    )
    writer.writeheader()
    writer.writerows(prediction_rows)
    atomic_write_text(predictions_out, predictions_buffer.getvalue())
    source_specs = sorted({spec for _, specs in requested for spec in specs
                           if view_archive(spec, args.task).exists()})
    manifest_out = out.with_name(f"{out.stem}_manifest.json")
    manifest = {
        "schema": "multiview_run_v1",
        "task": args.task,
        "complete": complete,
        "classes": TASKS[args.task]["classes"],
        "ordinal": TASKS[args.task]["ordinal"],
        "train_rows": len(train_keys),
        "test_rows": len(test_keys),
        "outlet_universe_sha256": outlet_universe_sha256(outlets),
        "rl_timesteps": args.rl_timesteps,
        "requested_rows": len(requested),
        "produced_rows": len(rows),
        "skipped": skipped,
        "metrics": rows,
        "sources": {
            spec: {
                "path": str(view_archive(spec, args.task).resolve()),
                "sha256": sha256_file(view_archive(spec, args.task)),
            }
            for spec in source_specs
        },
        "outputs": {
            "metrics": {"path": str(out.resolve()), "sha256": sha256_file(out)},
            "predictions": {
                "path": str(predictions_out.resolve()),
                "sha256": sha256_file(predictions_out),
            },
        },
    }
    atomic_write_text(manifest_out, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    print(f"wrote {predictions_out} and {manifest_out}; complete={complete}")
    for row in rows:
        print(f"{row['experiment']:70s} acc={row['accuracy']:6.2f} "
              f"macroF1={row['macro_f1']:6.2f} mae={row['mae']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--rl-timesteps", type=int, default=RL_TIMESTEPS)
    parser.add_argument("--only", default="",
                        help="comma-separated method filter, e.g. svm,rl")
    parser.add_argument("--allow-partial", action="store_true",
                        help="skip unavailable rows and write a clearly marked partial report")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.rl_timesteps < 0:
        parser.error("--rl-timesteps cannot be negative")
    try:
        with exclusive_file_lock(GRAPHS_DIR / f"fuse_{args.task}.lock"):
            run_benchmark(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
