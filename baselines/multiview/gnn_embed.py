"""Unsupervised GNN embeddings for the three media graphs (paper §5.1).

Paper-specified architecture: 64-dimensional embeddings per graph via GCN (GraphConv),
GraphSAGE, and ResGatedGCN, trained with a contrastive objective in an
unsupervised setting — epochs 50, layers 4, hidden 128, batch 128, lr 1e-4,
dropout 0.5. The contrastive method itself is unnamed in the paper; this
reimplementation uses Deep Graph Infomax. It shuffles informative node features
for Alexa and rewires edges for the constant-feature graphs.

Node features: Alexa graph = the 5 shipped Alexa attributes; hyperlink and
LLM graphs = a dummy constant feature (their Limitations: "GNNs were trained
using dummy features (e.g., a single integer per node)").

Labels are never touched. The graph encoder is transductive: train and test
nodes can both occur in the unlabeled topology. After training, every node is
encoded and matched outlets take their node's 64-d vector; unmatched outlets
get zero vectors (the paper's missing-view handling).

Output: data/graphs/emb_<graph>_<encoder>.npz keyed by outlet key.

Run all nine (graph x encoder) archives:
  /opt/anaconda3/bin/python3 baselines/multiview/gnn_embed.py --graph all --encoder all
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    EMB_DIM_GNN,
    GRAPHS_DIR,
    SOURCE_SHA256,
    emb_path,
    exclusive_file_lock,
    load_outlets,
    match_path,
    outlet_universe_sha256,
    save_embeddings,
    sha256_file,
)
from llm_graph import MAX_LEVEL, PROMPT_SHA256, RESPONSES_JSONL  # noqa: E402
from fetch_graphs import (  # noqa: E402
    ALEXA_EDGES,
    ALEXA_NODES,
    HYPERLINK_EDGES,
    HYPERLINK_NAME2ID,
    load_alexa_nodes,
    load_edge_csv,
)

ENCODERS = ("gcn", "sage", "resgated")
GRAPHS = ("alexa", "hyperlink", "llm")
HIDDEN, LAYERS, EPOCHS, BATCH, LR, DROPOUT = 128, 4, 50, 128, 1e-4, 0.5
SEED = 42
FULL_BATCH_MAX_NODES = 600_000  # current pinned graphs fit full-batch on CPU


def validate_pinned_graph_sources(name):
    expected = {
        "alexa": ((ALEXA_EDGES, "alexa_edges"), (ALEXA_NODES, "alexa_nodes")),
        "hyperlink": (
            (HYPERLINK_EDGES, "hyperlink_edges"),
            (HYPERLINK_NAME2ID, "hyperlink_name2id"),
        ),
    }
    for path, source_name in expected.get(name, ()):
        if not path.exists() or sha256_file(path) != SOURCE_SHA256[source_name]:
            raise ValueError(
                f"{path} is missing or differs from the pinned {source_name} source"
            )


def validate_llm_graph_manifest(outlets):
    manifest_path = GRAPHS_DIR / "llm_graph_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected_fields = {
        "schema": "multiview_llm_graph_v1",
        "complete": True,
        "max_level": MAX_LEVEL,
        "prompt_sha256": PROMPT_SHA256,
        "outlet_universe_sha256": outlet_universe_sha256(outlets),
    }
    for field, expected_value in expected_fields.items():
        if manifest.get(field) != expected_value:
            raise ValueError(
                f"{manifest_path}: {field}={manifest.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    outputs = manifest.get("outputs", {})
    expected_outputs = {
        "edges": GRAPHS_DIR / "llm_edges.csv",
        "name2id": GRAPHS_DIR / "llm_name2id.json",
        "matches": match_path("llm"),
        "responses": RESPONSES_JSONL,
    }
    for name, path in expected_outputs.items():
        record = outputs.get(name, {})
        if not path.exists() or record.get("sha256") != sha256_file(path):
            raise ValueError(f"{manifest_path}: invalid or stale {name} output")
    return manifest


def load_graph(name):
    """(x, edge_index, n_nodes) for one graph, undirected."""
    from torch_geometric.utils import to_undirected

    if name == "alexa":
        validate_pinned_graph_sources(name)
        domains, features = load_alexa_nodes()
        x = torch.tensor(features, dtype=torch.float32)
        edges = load_edge_csv(ALEXA_EDGES, n_nodes=len(domains))
    elif name == "hyperlink":
        validate_pinned_graph_sources(name)
        name2id = json.loads(HYPERLINK_NAME2ID.read_text())
        node_ids = {int(value) for value in name2id.values()}
        n = max(node_ids) + 1
        if len(node_ids) != len(name2id) or min(node_ids) != 0 or len(node_ids) != n:
            raise ValueError(f"{HYPERLINK_NAME2ID}: node ids are not unique and contiguous")
        x = torch.ones((n, 1), dtype=torch.float32)  # dummy feature, as in paper
        edges = load_edge_csv(HYPERLINK_EDGES, n_nodes=n)
    elif name == "llm":
        name2id_path = GRAPHS_DIR / "llm_name2id.json"
        edges_path = GRAPHS_DIR / "llm_edges.csv"
        if not name2id_path.exists() or not edges_path.exists():
            raise SystemExit(
                "LLM graph not generated yet — run llm_graph.py first "
                "(needs OPENAI_API_KEY) or skip --graph llm."
            )
        name2id = json.loads(name2id_path.read_text())
        node_ids = {int(value) for value in name2id.values()}
        if not node_ids:
            raise ValueError(f"{name2id_path}: empty node mapping")
        n = max(node_ids) + 1
        if len(node_ids) != len(name2id) or min(node_ids) != 0 or len(node_ids) != n:
            raise ValueError(f"{name2id_path}: node ids are not unique and contiguous")
        x = torch.ones((n, 1), dtype=torch.float32)  # dummy feature, as in paper
        edges = load_edge_csv(edges_path, n_nodes=n)
    else:
        raise ValueError(f"unknown graph {name!r}")
    if not edges:
        raise ValueError(f"{name} graph has no edges")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=x.shape[0])
    return x, edge_index, x.shape[0]


class Encoder(nn.Module):
    """4-layer GNN, hidden 128 -> 64-d output, dropout 0.5 (paper spec)."""

    def __init__(self, kind, in_dim, hidden=HIDDEN, out_dim=EMB_DIM_GNN,
                 layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        from torch_geometric.nn import GraphConv, ResGatedGraphConv, SAGEConv
        conv = {"gcn": GraphConv, "sage": SAGEConv, "resgated": ResGatedGraphConv}[kind]
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.convs = nn.ModuleList(conv(dims[i], dims[i + 1]) for i in range(layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x


def corruption(x, edge_index):
    """Shuffle informative features; corrupt topology for constant features."""
    if x.shape[0] > 1 and not torch.equal(x, x[:1].expand_as(x)):
        permutation = torch.randperm(x.shape[0], device=x.device)
        identity = torch.arange(x.shape[0], device=x.device)
        if torch.equal(permutation, identity):
            permutation = torch.roll(permutation, 1)
        return x[permutation], edge_index
    n_edges = edge_index.shape[1]
    if n_edges < 2:
        return x, edge_index
    permutation = torch.randperm(n_edges, device=edge_index.device)
    identity = torch.arange(n_edges, device=edge_index.device)
    if torch.equal(permutation, identity):
        permutation = torch.roll(permutation, 1)
    corrupted = torch.stack((edge_index[0], edge_index[1, permutation]))
    return x, corrupted


def summary_fn(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))


def train_dgi(kind, x, edge_index, epochs, device, batch=BATCH, log_every=10):
    from torch_geometric.data import Data
    from torch_geometric.nn import DeepGraphInfomax

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = DeepGraphInfomax(
        hidden_channels=EMB_DIM_GNN,
        encoder=Encoder(kind, x.shape[1]),
        summary=summary_fn,
        corruption=corruption,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_nodes = x.shape[0]

    if n_nodes <= FULL_BATCH_MAX_NODES:
        x_dev, ei_dev = x.to(device), edge_index.to(device)
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(x_dev, ei_dev)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            if epoch % log_every == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  loss {loss.item():.4f}", flush=True)
    else:  # neighbor-sampled fallback for future graphs larger than the pinned data
        from torch_geometric.loader import NeighborLoader
        data = Data(x=x, edge_index=edge_index)
        loader = NeighborLoader(data, num_neighbors=[10] * LAYERS,
                                batch_size=batch * 8, shuffle=True,
                                num_workers=0)
        for epoch in range(1, epochs + 1):
            model.train()
            total, count = 0.0, 0
            for sub in loader:
                sub = sub.to(device)
                optimizer.zero_grad()
                pos_z, neg_z, summary = model(sub.x, sub.edge_index)
                loss = model.loss(pos_z[:sub.batch_size], neg_z[:sub.batch_size], summary)
                loss.backward()
                optimizer.step()
                total += loss.item()
                count += 1
            if epoch % max(1, log_every // 2) == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{epochs}  mean loss {total / max(1, count):.4f}",
                      flush=True)
    return model


@torch.no_grad()
def encode_all(model, x, edge_index, device):
    model.eval()
    # Full-graph inference on CPU keeps memory deterministic for 528k nodes.
    cpu_model = model.to("cpu")
    z = cpu_model.encoder(x, edge_index)
    model.to(device)
    return z.numpy().astype(np.float32)


def embedding_output_path(graph, encoder, epochs):
    canonical = emb_path(graph, encoder=encoder)
    if epochs == EPOCHS:
        return canonical
    return canonical.with_name(f"{canonical.stem}_epochs{epochs}{canonical.suffix}")


def contrast_name(graph):
    return ("dgi_feature_shuffle_v1" if graph == "alexa"
            else "dgi_edge_destination_permutation_v1")


def graph_source_fingerprint(graph, match_file):
    if graph == "alexa":
        paths = [ALEXA_EDGES, ALEXA_NODES, match_file]
    elif graph == "hyperlink":
        paths = [HYPERLINK_EDGES, HYPERLINK_NAME2ID, match_file]
    else:
        paths = [
            GRAPHS_DIR / "llm_edges.csv",
            GRAPHS_DIR / "llm_name2id.json",
            GRAPHS_DIR / "llm_graph_manifest.json",
            RESPONSES_JSONL,
            match_file,
        ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing graph source files: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def embed_graph(graph, encoder, epochs, device, outlets):
    print(f"=== {graph} / {encoder} ===")
    if graph == "llm":
        validate_llm_graph_manifest(outlets)
    x, edge_index, n_nodes = load_graph(graph)
    print(f"nodes={n_nodes}  directed edge-index entries={edge_index.shape[1]}  "
          f"features={x.shape[1]}")

    match_file = match_path(graph)
    matches = json.loads(match_file.read_text())
    keys = sorted(outlets)
    unknown = sorted(set(matches) - set(keys))
    if unknown:
        raise ValueError(
            f"{match_file} contains {len(unknown)} outlets outside the canonical universe"
        )
    if len(matches.values()) != len(set(int(value) for value in matches.values())):
        raise ValueError(f"{match_file} maps multiple outlets to one graph node")
    invalid_node_ids = sorted({int(value) for value in matches.values()
                               if not 0 <= int(value) < n_nodes})
    if invalid_node_ids:
        raise ValueError(f"{match_file} contains out-of-range node ids: {invalid_node_ids[:10]}")
    source_sha256 = graph_source_fingerprint(graph, match_file)

    model = train_dgi(encoder, x, edge_index, epochs, device)
    z = encode_all(model, x, edge_index, device)
    X = np.zeros((len(keys), EMB_DIM_GNN), dtype=np.float32)
    matched = 0
    for i, key in enumerate(keys):
        node_id = matches.get(key)
        if node_id is not None and 0 <= int(node_id) < n_nodes:
            X[i] = z[int(node_id)]
            matched += 1
    out = embedding_output_path(graph, encoder, epochs)
    save_embeddings(
        out, keys, X,
        schema="multiview_embedding_v2",
        complete=bool(epochs == EPOCHS),
        graph=graph,
        encoder=encoder,
        epochs=epochs,
        contrast=contrast_name(graph),
        n_nodes=n_nodes,
        matched_outlets=matched,
        outlet_universe_sha256=outlet_universe_sha256(outlets),
        match_sha256=sha256_file(match_file),
        graph_source_sha256=source_sha256,
    )
    print(
        f"saved {out}  X{X.shape}  matched={matched}/{len(keys)}"
    )


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    # PyG scatter kernels are unreliable on MPS; CPU is fast enough here.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=list(GRAPHS) + ["all"], required=True)
    parser.add_argument("--encoder", choices=list(ENCODERS) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    device = pick_device(args.device)
    print(f"device: {device}")

    outlets = load_outlets()
    graphs = list(GRAPHS) if args.graph == "all" else [args.graph]
    encoders = list(ENCODERS) if args.encoder == "all" else [args.encoder]
    for graph in graphs:
        for encoder in encoders:
            output = embedding_output_path(graph, encoder, args.epochs)
            try:
                with exclusive_file_lock(output.with_suffix(output.suffix + ".lock")):
                    embed_graph(graph, encoder, args.epochs, device, outlets)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise SystemExit(
                    f"cannot build {graph}/{encoder}: {exc}"
                ) from None


if __name__ == "__main__":
    main()
