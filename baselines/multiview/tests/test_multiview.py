"""Unit tests for the Multi-View suite reconstruction (no network, no API)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared
import llm_graph
from fetch_graphs import extract_hyperlink_zip, load_edge_csv


# -------------------------------------------------------------- shared ----

def test_registered_domain_variants():
    assert shared.registered_domain("https://www.nytimes.com/section/world") == "nytimes.com"
    assert shared.registered_domain("nytimes.com") == "nytimes.com"
    assert shared.registered_domain("http://news.bbc.co.uk") == "bbc.co.uk"
    assert shared.registered_domain("HTTPS://Example.ORG:8080/x") == "example.org"
    assert shared.registered_domain("") is None
    assert shared.registered_domain(None) is None
    assert shared.normalized_host("https://News.Example.org:443/path") == "news.example.org"


def test_candidate_exact_hosts_uses_mbfc_alias_in_order():
    outlet = {"url": "https://9news.com.au", "media_name": "Nine News"}
    mbfc_by_name = {"nine news": {"domain": "nine.com.au"}}
    hosts = shared.candidate_exact_hosts(outlet, mbfc_by_name)
    assert hosts == ("9news.com.au", "nine.com.au")


def test_candidate_node_keys_use_mbfc_domain_not_review_url():
    outlet = {"url": "https://outlet.example", "media_name": "Outlet"}
    aliases = {"outlet": {
        "domain": "real-outlet.com",
        "url": "https://mediabiasfactcheck.com/outlet-review/",
    }}
    keys = shared.candidate_node_keys(outlet, aliases)
    assert "@host:real-outlet.com" in keys
    assert all("mediabiasfactcheck.com" not in key for key in keys)


def test_unique_node_index_does_not_collapse_hosting_subdomains():
    index = shared.unique_node_host_index({
        "foo.wordpress.com": 1,
        "bar.wordpress.com": 2,
        "wordpress.com": 3,
        "cnn.com": 4,
    })
    assert index["@host:foo.wordpress.com"] == 1
    assert index["@host:bar.wordpress.com"] == 2
    assert index["@host:wordpress.com"] == 3
    assert "@registered:wordpress.com" not in index
    assert index["@host:cnn.com"] == 4
    unmatched = shared.match_outlets_to_nodes(
        {"site": {"url": "https://baz.wordpress.com", "media_name": "Baz"}},
        index,
    )
    assert unmatched == {}


def test_match_outlets_prefers_primary_domain():
    outlets = {"a": {"url": "https://foo.com", "media_name": "Foo"},
               "b": {"url": "https://bar.com", "media_name": "Bar"}}
    node_ids = shared.unique_node_host_index({"https://foo.com": 7})
    matches = shared.match_outlets_to_nodes(outlets, node_ids)
    assert matches == {"a": 7}


def test_duplicate_graph_nodes_keep_test_outlet():
    outlets = {
        "train": {"split": "train", "label_3class": "LOW", "bias_rating": "LEFT"},
        "test": {"split": "test", "label_3class": "HIGH", "bias_rating": "RIGHT"},
    }
    matches, dropped = shared.deduplicate_node_matches(
        {"train": 4, "test": 4}, outlets,
    )
    assert matches == {"test": 4}
    assert dropped == [{"key": "train", "kept": "test", "node_id": 4}]


def test_embeddings_roundtrip(tmp_path):
    path = tmp_path / "emb.npz"
    keys = ["k1", "k2", "k3"]
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    shared.save_embeddings(path, keys, X, view="test")
    keys2, X2 = shared.load_embeddings(path, expected_keys=keys)
    assert keys2 == keys
    np.testing.assert_array_equal(X, X2)
    with pytest.raises(ValueError):
        shared.load_embeddings(path, expected_keys=["k1", "k2", "WRONG"])


def test_exclusive_file_lock_rejects_second_owner(tmp_path):
    lock_path = tmp_path / "job.lock"
    with shared.exclusive_file_lock(lock_path):
        with pytest.raises(RuntimeError, match="another run"):
            with shared.exclusive_file_lock(lock_path):
                pass


def test_load_embeddings_rejects_nonfinite(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, keys=np.array(["k"]), X=np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ValueError):
        shared.load_embeddings(path)
    with pytest.raises(ValueError):
        shared.save_embeddings(path, ["k"], np.array([[np.nan]], dtype=np.float32))


def test_embedding_metadata_is_required_when_requested(tmp_path):
    path = tmp_path / "emb.npz"
    shared.save_embeddings(path, ["k"], np.ones((1, 2)), schema="v2", complete=True)
    shared.load_embeddings(
        path, expected_keys=["k"],
        expected_metadata={"schema": "v2", "complete": True},
    )
    with pytest.raises(ValueError, match="metadata"):
        shared.load_embeddings(path, expected_metadata={"schema": "old"})


def test_duplicate_nonzero_embeddings_prefer_test_and_zero_train():
    outlets = {
        "a": {"split": "train", "label_3class": "LOW", "bias_rating": "LEFT"},
        "b": {"split": "test", "label_3class": "LOW", "bias_rating": "LEFT"},
        "c": {"split": "train", "label_3class": "HIGH", "bias_rating": "RIGHT"},
    }
    X = np.array([[1, 2], [1, 2], [3, 4]], dtype=np.float32)
    clean, dropped = shared.deduplicate_embedding_rows(["a", "b", "c"], X, outlets)
    np.testing.assert_array_equal(clean[0], np.zeros(2))
    np.testing.assert_array_equal(clean[1], X[1])
    assert dropped == [{"key": "a", "kept": "b"}]


def test_load_outlets_uses_only_canonical_clean_rows(tmp_path, monkeypatch):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    header = "media_name,url,image_path,label_3class,bias_rating\n"
    train.write_text(header + "Keep,https://keep.com,Keep.png,LOW,LEFT\n")
    test.write_text(header + "Held,https://held.com,Held.png,HIGH,RIGHT\n")
    raw = {
        "keep": {"key": "keep", "n_articles": 1, "articles": [{"text": "a"}]},
        "held": {"key": "held", "n_articles": 1, "articles": [{"text": "b"}]},
        "removed": {"key": "removed", "n_articles": 1, "articles": []},
    }
    monkeypatch.setattr(shared, "TRAIN_CSV", train)
    monkeypatch.setattr(shared, "TEST_CSV", test)
    monkeypatch.setattr(shared.mgm_pipeline, "load_outlets", lambda: raw)
    outlets = shared.load_outlets()
    assert set(outlets) == {"keep", "held"}
    assert outlets["keep"]["split"] == "train"
    assert outlets["held"]["split"] == "test"


# ------------------------------------------------------- fetch_graphs -----

def test_load_edge_csv_bounds(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("source,target\n0,1\n1,2\n")
    assert load_edge_csv(path, n_nodes=3) == [(0, 1), (1, 2)]
    with pytest.raises(SystemExit):
        load_edge_csv(path, n_nodes=2)


def test_extract_hyperlink_zip_tsv(tmp_path):
    import zipfile
    zpath = tmp_path / "graph.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("graph.tsv", "0\t1\n1\t2\n")
    out = extract_hyperlink_zip(zpath, tmp_path / "edges.csv")
    assert load_edge_csv(out) == [(0, 1), (1, 2)]


def test_extract_hyperlink_zip_with_header(tmp_path):
    import zipfile
    zpath = tmp_path / "graph.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("graph.csv", "source,target\n3,4\n")
    out = extract_hyperlink_zip(zpath, tmp_path / "edges.csv")
    assert load_edge_csv(out) == [(3, 4)]


def test_unique_node_host_index_url_host_and_domain_keys():
    ids = shared.unique_node_host_index({"https://www.foo.com/": 0, "bar.co.uk": 1})
    assert ids["@url:https://www.foo.com"] == 0
    assert ids["@host:foo.com"] == 0
    assert ids["@host:bar.co.uk"] == 1
    assert ids["@registered:foo.com"] == 0


# ----------------------------------------------------------- llm_graph ----

def test_parse_similar_dedup_and_self():
    text = ("<s> https://www.cnn.com/ </s>\n<s>cnn.com</s>\n"
            "<s> reuters.com </s><s>https://self.com</s>")
    assert llm_graph.parse_similar(text, self_domain="self.com") == ["cnn.com", "reuters.com"]


def test_parse_similar_garbage():
    assert llm_graph.parse_similar("no tags here") == []
    assert llm_graph.parse_similar(None) == []


def test_parse_similar_caps_response_at_five():
    response = "".join(f"<s>site{i}.com</s>" for i in range(8))
    assert llm_graph.parse_similar(response) == [f"site{i}.com" for i in range(5)]


def test_expand_graph_levels_and_resume(tmp_path):
    responses = tmp_path / "resp.jsonl"
    querier = llm_graph.DryRunQuerier()
    seeds = ["cnn.com", "nytimes.com"]
    cache = {}
    level_of, edges, complete = llm_graph.expand_graph(
        seeds, querier, cache, max_level=3, responses_path=responses)
    assert complete
    assert level_of["cnn.com"] == 0 and level_of["nytimes.com"] == 0
    assert all(lvl <= 3 for lvl in level_of.values())
    assert edges
    n_queries_first = sum(1 for _ in open(responses))
    # resume: nothing new should be queried
    cache2 = llm_graph.load_cache(responses)
    assert set(cache2) == set(cache)
    level_of2, edges2, complete2 = llm_graph.expand_graph(
        seeds, querier, cache2, max_level=3, responses_path=responses)
    assert (level_of2, edges2) == (level_of, edges)
    assert complete2
    assert sum(1 for _ in open(responses)) == n_queries_first


def test_expand_graph_max_queries_cap(tmp_path):
    responses = tmp_path / "resp.jsonl"
    level_of, _, complete = llm_graph.expand_graph(
        ["a.com", "b.com", "c.com"], llm_graph.DryRunQuerier(), {},
        max_level=3, max_queries=1, responses_path=responses)
    assert not complete
    assert sum(1 for _ in open(responses)) == 1


def test_llm_cache_rejects_mixed_models(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        '{"domain":"cnn.com","model":"old","similar":["reuters.com"]}\n'
    )
    with pytest.raises(ValueError, match="cached model"):
        llm_graph.load_cache(path, expected_model="new")


def test_llm_cache_tolerates_only_interrupted_final_line(tmp_path):
    path = tmp_path / "responses.jsonl"
    good = '{"domain":"cnn.com","model":"m","similar":["reuters.com"]}\n'
    path.write_text(good + '{"domain":')
    assert llm_graph.load_cache(path, expected_model="m") == {
        "cnn.com": ["reuters.com"],
    }
    path.write_text('{"domain":\n' + good)
    with pytest.raises(ValueError, match="malformed JSON"):
        llm_graph.load_cache(path, expected_model="m")
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="not an object"):
        llm_graph.load_cache(path)
    path.write_text(json.dumps({
        "domain": "cnn.com", "model": "m",
        "similar": [f"site{i}.com" for i in range(6)],
    }) + "\n")
    with pytest.raises(ValueError, match="invalid similar-site list"):
        llm_graph.load_cache(path)


def test_llm_seed_domains_uses_one_candidate_per_outlet():
    outlets = {
        "a": {"url": "https://a.com", "media_name": "A"},
        "b": {"url": "https://a.com", "media_name": "B"},
    }
    aliases = {
        "a": {"domain": "alias-a.com"},
        "b": {"domain": "alias-b.com"},
    }
    assert llm_graph.seed_domains(outlets, aliases) == ["a.com"]


def test_llm_nonstandard_outputs_are_suffixed():
    assert llm_graph.output_suffix(False, 0, llm_graph.MAX_LEVEL) == ""
    assert llm_graph.output_suffix(False, 10, llm_graph.MAX_LEVEL) == "_seeds10"
    assert llm_graph.output_suffix(True, 10, 2) == "_dryrun_seeds10_level2"


def test_llm_dryrun_outputs_are_suffixed_and_manifested(tmp_path, monkeypatch):
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        '{"domain":"a.com","model":"dry-run","similar":["b.com"]}\n'
    )
    monkeypatch.setattr(llm_graph, "GRAPHS_DIR", tmp_path)
    monkeypatch.setattr(llm_graph, "LLM_EDGES", tmp_path / "llm_edges.csv")
    monkeypatch.setattr(llm_graph, "LLM_NAME2ID", tmp_path / "llm_name2id.json")
    monkeypatch.setattr(
        llm_graph, "match_path", lambda view: tmp_path / "graph_match_llm.json",
    )
    outlets = {
        "a": {
            "url": "https://a.com", "media_name": "A", "split": "train",
            "label_3class": "LOW", "bias_rating": "LEFT",
        }
    }
    llm_graph.build_outputs(
        {"a.com": 0, "b.com": 1}, {("a.com", "b.com")}, outlets, {},
        "dry-run", 3, responses, suffix="_dryrun",
    )
    assert (tmp_path / "llm_edges_dryrun.csv").exists()
    assert (tmp_path / "graph_match_llm_dryrun.json").exists()
    manifest = json.loads((tmp_path / "llm_graph_manifest_dryrun.json").read_text())
    assert manifest["complete"] is True
    assert manifest["model"] == "dry-run"


def test_openai_querier_requires_env_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        llm_graph.OpenAIQuerier("gpt-3.5-turbo-0125")


# ----------------------------------------------------------- gnn_embed ----

def test_encoder_shapes_all_kinds():
    import torch
    from gnn_embed import Encoder
    x = torch.randn(10, 5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    for kind in ("gcn", "sage", "resgated"):
        z = Encoder(kind, 5)(x, edge_index)
        assert z.shape == (10, shared.EMB_DIM_GNN)
        assert torch.isfinite(z).all()


def test_dgi_trains_on_toy_graph():
    import torch
    from gnn_embed import encode_all, train_dgi
    torch.manual_seed(0)
    x = torch.randn(30, 5)
    src = torch.arange(29)
    edge_index = torch.stack([src, src + 1])
    model = train_dgi("gcn", x, edge_index, epochs=2, device=torch.device("cpu"))
    z = encode_all(model, x, edge_index, torch.device("cpu"))
    assert z.shape == (30, shared.EMB_DIM_GNN)
    assert np.isfinite(z).all()


def test_dgi_corruption_changes_graph_with_constant_features():
    import torch
    from gnn_embed import corruption
    x = torch.ones((5, 1))
    edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 4]])
    corrupt_x, corrupt_edges = corruption(x, edge_index)
    assert torch.equal(corrupt_x, x)
    assert torch.equal(corrupt_edges[0], edge_index[0])
    assert not torch.equal(corrupt_edges, edge_index)


def test_constant_feature_corruption_changes_encoder_output():
    import torch
    from gnn_embed import Encoder, corruption
    x = torch.ones((20, 1))
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ])
    _, corrupt_edges = corruption(x, edge_index)
    for kind in ("gcn", "sage", "resgated"):
        torch.manual_seed(42)
        encoder = Encoder(kind, 1).eval()
        assert not torch.equal(encoder(x, edge_index), encoder(x, corrupt_edges))


def test_dgi_corruption_shuffles_informative_features():
    import torch
    from gnn_embed import corruption
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    corrupt_x, corrupt_edges = corruption(x, edge_index)
    assert not torch.equal(corrupt_x, x)
    assert torch.equal(corrupt_edges, edge_index)


def test_graph_smoke_epochs_do_not_use_canonical_path():
    from gnn_embed import EPOCHS, embedding_output_path
    assert embedding_output_path("alexa", "gcn", EPOCHS).name == "emb_alexa_gcn.npz"
    assert embedding_output_path("alexa", "gcn", 1).name == "emb_alexa_gcn_epochs1.npz"


# ------------------------------------------------------ text_embed -----

def test_accumulation_divisor_handles_short_final_group():
    from text_embed import accumulation_divisor
    assert [accumulation_divisor(step, 8, 6) for step in range(1, 9)] == [
        6, 6, 6, 6, 6, 6, 2, 2,
    ]


def test_wiki_duplicates_reject_conflicts_and_keep_one_copy(tmp_path, monkeypatch):
    import json
    import text_embed
    path = tmp_path / "wiki.jsonl"
    records = [
        {"key": "a", "found": True, "text": "same conflict"},
        {"key": "b", "found": True, "text": "same   conflict"},
        {"key": "c", "found": True, "text": "same label"},
        {"key": "d", "found": True, "text": "same label"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    outlets = {
        "a": {"split": "train", "bias_rating": "LEFT"},
        "b": {"split": "test", "bias_rating": "RIGHT"},
        "c": {"split": "train", "bias_rating": "RIGHT"},
        "d": {"split": "test", "bias_rating": "RIGHT"},
    }
    monkeypatch.setattr(text_embed, "WIKI_PAGES", path)
    texts, report = text_embed.load_wiki_texts(outlets, "bias")
    assert texts == {"d": "same label"}
    assert report["duplicate_groups"] == 2
    assert report["conflicting_label_groups"] == 1


def test_wiki_rejects_conflicting_records_for_same_key(tmp_path, monkeypatch):
    import text_embed
    path = tmp_path / "wiki.jsonl"
    path.write_text(
        '{"key":"a","found":true,"text":"first"}\n'
        '{"key":"a","found":true,"text":"second"}\n'
    )
    monkeypatch.setattr(text_embed, "WIKI_PAGES", path)
    with pytest.raises(ValueError, match="conflicting duplicate records"):
        text_embed.load_wiki_texts({"a": {"bias_rating": "LEFT"}}, "bias")


def test_text_smoke_limits_do_not_use_canonical_path():
    from text_embed import EPOCHS, embedding_output_path
    assert embedding_output_path("wiki", "bias", EPOCHS, 0).name == "emb_wiki_bias.npz"
    assert embedding_output_path("wiki", "bias", 1, 20).name == (
        "emb_wiki_bias_limit20_epochs1.npz"
    )


# ---------------------------------------------------------------- fuse ----

def _toy_problem(n_per_class=40, n_classes=3, dims=(8, 6), seed=0):
    rng = np.random.default_rng(seed)
    labels, views = [], [[] for _ in dims]
    classes = ["LOW", "MIXED", "HIGH"][:n_classes]
    for ci, c in enumerate(classes):
        for _ in range(n_per_class):
            labels.append(c)
            for vi, d in enumerate(dims):
                center = np.zeros(d)
                center[ci % d] = 3.0
                views[vi].append(center + rng.normal(0, 0.5, d))
    y = list(labels)
    X = [np.array(v, dtype=np.float32) for v in views]
    return X, y, classes


def test_svm_fusion_learns_toy():
    from fuse import fit_predict_svm
    X, y, _ = _toy_problem()
    Xc = np.concatenate(X, axis=1)
    pred = fit_predict_svm(Xc[::2], y[::2], Xc[1::2])
    acc = np.mean([p == t for p, t in zip(pred, y[1::2])])
    assert acc > 0.9


def test_standardize_keeps_missing_rows_zero():
    from fuse import standardize
    fit = np.array([[0, 0], [2, 4], [4, 8]], dtype=np.float32)
    test = np.array([[0, 0], [3, 6]], dtype=np.float32)
    fit_scaled, test_scaled = standardize(fit, test)
    np.testing.assert_array_equal(fit_scaled[0], np.zeros(2))
    np.testing.assert_array_equal(test_scaled[0], np.zeros(2))


def test_view_projection_keeps_missing_token_zero():
    import torch
    from fuse import ViewTokens
    tokens = ViewTokens([3, 4])([
        torch.zeros((2, 3)), torch.zeros((2, 4)),
    ])
    assert torch.equal(tokens, torch.zeros_like(tokens))


def test_paper_rows_are_task_specific_and_attention_groups_are_valid():
    import fuse
    bias = fuse.benchmark_rows("bias")
    factuality = fuse.benchmark_rows("factuality")
    assert ("mlp", ["llm_gcn", "hyperlink_sage", "articles", "wiki"]) in bias
    assert ("co_attn", ["alexa_resgated", "articles", "wiki"]) in factuality
    assert not any(method == "co_attn" for method, _ in bias)
    assert fuse.text_first_query_idx(["articles", "wiki"]) == [0]
    assert fuse.text_first_query_idx(["llm_gcn", "articles", "wiki"]) == [1, 2]


def test_fusion_rejects_stale_graph_match_fingerprint(tmp_path, monkeypatch):
    import fuse
    archive = tmp_path / "emb_alexa_gcn.npz"
    shared.save_embeddings(
        archive, ["a"], np.ones((1, shared.EMB_DIM_GNN)),
        schema="multiview_embedding_v2", complete=True,
        graph="alexa", encoder="gcn", epochs=50,
        contrast="dgi_feature_shuffle_v1",
        outlet_universe_sha256="universe",
        match_sha256="old-match", graph_source_sha256="source",
    )
    monkeypatch.setattr(fuse, "view_archive", lambda spec, task: archive)
    monkeypatch.setattr(fuse, "current_source_sha256", lambda path: "new-match")
    monkeypatch.setattr(fuse, "current_graph_fingerprint", lambda graph: "source")
    with pytest.raises(ValueError, match="match_sha256"):
        fuse.load_views(["alexa_gcn"], "factuality", ["a"], "universe")


@pytest.mark.parametrize("method", ["mlp", "self_attn", "cross_attn", "co_attn"])
def test_torch_fusions_learn_toy(method):
    from fuse import (CrossAttnFusion, MLPFusion, SelfAttnFusion,
                      predict_torch_fusion, train_torch_fusion)
    X, y, classes = _toy_problem()
    label2id = {c: i for i, c in enumerate(classes)}
    yid = [label2id[c] for c in y]
    fit = [v[::2] for v in X]
    dev = [v[1::2] for v in X]
    yf, yd = yid[::2], yid[1::2]
    dims = [v.shape[1] for v in fit]
    if method == "mlp":
        model = MLPFusion(sum(dims), len(classes))
    elif method == "self_attn":
        model = SelfAttnFusion(dims, len(classes))
    else:
        model = CrossAttnFusion(dims, [0], len(classes), co=(method == "co_attn"))
    model, dev_f1 = train_torch_fusion(model, fit, yf, dev, yd, epochs=60, patience=60)
    assert dev_f1 > 0.85
    pred = predict_torch_fusion(model, dev)
    assert pred.shape == (len(yd),)


def test_rl_fusion_runs_and_beats_chance():
    from fuse import fit_predict_rl
    X, y, classes = _toy_problem(n_per_class=30)
    label2id = {c: i for i, c in enumerate(classes)}
    yid = [label2id[c] for c in y]
    fit = [v[::2] for v in X]
    dev = [v[1::2] for v in X]
    yf, yd = yid[::2], yid[1::2]
    pred_dev, pred_test = fit_predict_rl(fit, yf, dev, yd, dev, len(classes),
                                         timesteps=2048)
    acc = np.mean(np.asarray(pred_dev) == np.asarray(yd))
    assert acc > 1.0 / len(classes) + 0.2  # far above chance on separable toy
    assert pred_test.shape == (len(yd),)


def test_run_row_majority_and_svm_end_to_end(tmp_path, monkeypatch):
    """Synthetic outlets + embedding archives through the real run_row path."""
    import fuse

    keys = [f"o{i:02d}" for i in range(40)]
    classes = ["LOW", "MIXED", "HIGH"]
    outlets = {}
    rng = np.random.default_rng(1)
    for i, key in enumerate(keys):
        outlets[key] = {"key": key, "split": "train" if i < 30 else "test",
                        "label_3class": classes[i % 3], "bias_rating": "LEAST BIASED",
                        "n_articles": 1, "articles": []}
    X = np.zeros((len(keys), 768), dtype=np.float32)
    for i in range(len(keys)):
        X[i, i % 3] = 5.0
        X[i] += rng.normal(0, 0.01, 768)
    archive = tmp_path / "emb_articles_factuality.npz"
    universe_sha256 = shared.outlet_universe_sha256(outlets)
    shared.save_embeddings(
        archive, keys, X,
        schema="multiview_embedding_v2",
        complete=True,
        view="articles",
        task="factuality",
        model="distilbert-base-uncased",
        model_revision="12040accade4e8a0f71eabdb258fecc2e7e948be",
        epochs=6,
        max_len=256,
        source_sha256="toy-source",
        content_filter_sha256="toy-filter",
        outlet_universe_sha256=universe_sha256,
    )
    monkeypatch.setattr(fuse, "view_archive", lambda spec, task: archive)
    monkeypatch.setattr(fuse, "current_source_sha256", lambda path: "toy-source")
    monkeypatch.setattr(fuse, "current_article_filter_sha256", lambda: "toy-filter")

    train_keys = sorted(k for k in keys if outlets[k]["split"] == "train")
    test_keys = sorted(k for k in keys if outlets[k]["split"] == "test")
    labels = {k: shared.outlet_label(outlets[k], "factuality") for k in keys}

    row = fuse.run_row("majority", [], "factuality", outlets, train_keys,
                       test_keys, labels, rl_timesteps=0)
    assert row["n_test"] == len(test_keys)
    row = fuse.run_row("svm", ["articles"], "factuality", outlets, train_keys,
                       test_keys, labels, rl_timesteps=0)
    assert row["accuracy"] >= 90.0
    assert set(["accuracy", "balanced_accuracy", "macro_precision",
                "macro_recall", "macro_f1", "mae", "mse"]) <= set(row)


def test_dev_carve_disjoint_and_stratified():
    from fuse import dev_carve
    keys = [f"k{i}" for i in range(100)]
    labels = {k: ("A" if i % 2 else "B") for i, k in enumerate(keys)}
    fit, dev = dev_carve(keys, labels)
    assert not set(fit) & set(dev)
    assert sorted(fit + dev) == sorted(keys)
    dev_labels = [labels[k] for k in dev]
    assert dev_labels.count("A") == dev_labels.count("B")
