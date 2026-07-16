"""Download the Alexa + Hyperlink graphs and match our outlets onto them.

Alexa graph (paper's strongest view, §3.1): the frozen level-3 audience-overlap
crawl from Panayotov'22, republished as plain CSVs in marslanm/MGM_code —
Edge_level_3.csv (integer edges; ids are row indices of the node table) and
ACL_level_3_fact.csv (site domain + the 5 Alexa node attributes; we ignore the
shipped labels/masks and bring our own).

Hyperlink (on-site) graph: marslanm/multi-graph-perspective onsite_data/
MBFC-2023 — graph edges + name2id.json (domain -> node id). No node features
(the paper trains those GNNs on dummy features).

Output (data/graphs/, git-ignored):
- alexa_edges.csv, alexa_nodes.csv, hyperlink_edges.csv, hyperlink_name2id.json
- graph_match_alexa.json / graph_match_hyperlink.json: outlet key -> node id
- a coverage report per task/split on stdout.
"""

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

from shared import (
    ALEXA_EDGES_URL,
    ALEXA_FEATURE_COLUMNS,
    ALEXA_NODES_URL,
    GRAPHS_DIR,
    HYPERLINK_GRAPH_ZIP_URL,
    HYPERLINK_NAME2ID_URL,
    SOURCE_SHA256,
    TASKS,
    atomic_write_text,
    deduplicate_node_matches,
    download,
    exclusive_file_lock,
    load_mbfc_sources,
    load_outlets,
    match_outlets_to_nodes,
    match_path,
    mbfc_name_index,
    outlet_label,
    sha256_file,
    unique_node_host_index,
)

ALEXA_EDGES = GRAPHS_DIR / "alexa_edges.csv"
ALEXA_NODES = GRAPHS_DIR / "alexa_nodes.csv"
HYPERLINK_ZIP = GRAPHS_DIR / "hyperlink_graph.zip"
HYPERLINK_EDGES = GRAPHS_DIR / "hyperlink_edges.csv"
HYPERLINK_NAME2ID = GRAPHS_DIR / "hyperlink_name2id.json"


def load_alexa_nodes(path=ALEXA_NODES):
    """(domains, features) in row order; row index == edge-file node id."""
    domains, features = [], []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ALEXA_FEATURE_COLUMNS + ["site"] if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path} is missing expected columns: {missing}")
        for row in reader:
            domains.append(str(row["site"]).strip().lower())
            values = [float(row[c] or 0.0) for c in ALEXA_FEATURE_COLUMNS]
            if not all(math.isfinite(value) for value in values):
                raise SystemExit(f"{path}: non-finite Alexa features at row {reader.line_num}")
            features.append(values)
    return domains, features


def load_edge_csv(path, n_nodes=None, source_col="source", target_col="target"):
    edges = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source, target = int(row[source_col]), int(row[target_col])
            if n_nodes is not None and not (0 <= source < n_nodes and 0 <= target < n_nodes):
                raise SystemExit(
                    f"{path}: edge ({source},{target}) outside node table of size {n_nodes}")
            edges.append((source, target))
    return edges


def extract_hyperlink_zip(zip_path=HYPERLINK_ZIP, out_path=HYPERLINK_EDGES):
    """Pull the single edge table out of graph.zip into a plain CSV."""
    if out_path.exists() and out_path.stat().st_size > 100:
        if sha256_file(out_path) == SOURCE_SHA256["hyperlink_edges"]:
            return out_path
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist()
                 if not n.endswith("/") and not Path(n).name.startswith(".")]
        if not names:
            raise SystemExit(f"{zip_path} contains no data files")
        member = max(names, key=lambda n: archive.getinfo(n).file_size)
        payload = archive.read(member).decode("utf-8", errors="replace")
    # Normalize whatever separator ships (tsv/csv) into source,target CSV.
    rows = []
    reader = csv.reader(io.StringIO(payload),
                        delimiter="\t" if "\t" in payload.splitlines()[0] else ",")
    header = next(reader)
    has_header = not all(cell.strip().lstrip("-").isdigit() for cell in header[:2] if cell.strip())
    if not has_header:
        rows.append(header)
    rows.extend(reader)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["source", "target"])
    for row in rows:
        if len(row) >= 2 and row[0].strip() and row[1].strip():
            writer.writerow([row[0].strip(), row[1].strip()])
    atomic_write_text(out_path, output.getvalue())
    if zip_path == HYPERLINK_ZIP and sha256_file(out_path) != SOURCE_SHA256["hyperlink_edges"]:
        raise IOError(f"{out_path}: normalized hyperlink edge hash mismatch")
    return out_path


def coverage_report(view, matches, outlets):
    print(f"\n=== {view} graph coverage ===")
    print(f"matched outlets: {len(matches)}/{len(outlets)}")
    for task in TASKS:
        for split in ("train", "test"):
            labeled = [k for k, o in outlets.items()
                       if o.get("split") == split and outlet_label(o, task)]
            hit = sum(1 for k in labeled if k in matches)
            print(f"  {task:10s} {split:5s}: {hit:4d}/{len(labeled):4d} "
                  f"({hit / max(1, len(labeled)):.1%}) outlets have a node")


def run(args):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("downloading Alexa graph (MGM_code)...")
        download(
            ALEXA_EDGES_URL, ALEXA_EDGES, expect_min_bytes=10_000,
            expected_sha256=SOURCE_SHA256["alexa_edges"],
        )
        download(
            ALEXA_NODES_URL, ALEXA_NODES, expect_min_bytes=100_000,
            expected_sha256=SOURCE_SHA256["alexa_nodes"],
        )
        print("downloading Hyperlink graph (multi-graph-perspective)...")
        download(
            HYPERLINK_GRAPH_ZIP_URL, HYPERLINK_ZIP, expect_min_bytes=10_000,
            expected_sha256=SOURCE_SHA256["hyperlink_graph"],
        )
        download(
            HYPERLINK_NAME2ID_URL, HYPERLINK_NAME2ID, expect_min_bytes=1_000,
            expected_sha256=SOURCE_SHA256["hyperlink_name2id"],
        )
    else:
        local_sources = {
            ALEXA_EDGES: "alexa_edges",
            ALEXA_NODES: "alexa_nodes",
            HYPERLINK_ZIP: "hyperlink_graph",
            HYPERLINK_NAME2ID: "hyperlink_name2id",
        }
        for path, source_name in local_sources.items():
            if not path.exists() or sha256_file(path) != SOURCE_SHA256[source_name]:
                raise SystemExit(
                    f"{path} is missing or fails its pinned SHA-256; rerun without --skip-download"
                )
    extract_hyperlink_zip()

    alexa_domains, _ = load_alexa_nodes()
    alexa_edges = load_edge_csv(ALEXA_EDGES, n_nodes=len(alexa_domains))
    print(f"Alexa graph: {len(alexa_domains)} nodes, {len(alexa_edges)} edges")

    name2id = json.loads(HYPERLINK_NAME2ID.read_text())
    hyper_by_domain = unique_node_host_index(name2id)
    hyper_edges = load_edge_csv(HYPERLINK_EDGES)
    print(f"Hyperlink graph: {len(name2id)} named nodes "
          f"({len(hyper_by_domain)} safe host lookups), "
          f"{len(hyper_edges)} directed adjacency rows "
          "(1,994,492 unique undirected edges in the pinned source)")

    outlets = load_outlets()
    mbfc = mbfc_name_index(load_mbfc_sources())

    alexa_by_domain = unique_node_host_index(
        {domain: node_id for node_id, domain in enumerate(alexa_domains)}
    )
    for view, by_domain in (("alexa", alexa_by_domain), ("hyperlink", hyper_by_domain)):
        raw_matches = match_outlets_to_nodes(outlets, by_domain, mbfc)
        matches, dropped = deduplicate_node_matches(raw_matches, outlets)
        atomic_write_text(match_path(view), json.dumps(matches, indent=0, sort_keys=True))
        dropped_path = GRAPHS_DIR / f"graph_match_{view}_dropped.json"
        atomic_write_text(
            dropped_path, json.dumps(dropped, indent=2, sort_keys=True) + "\n",
        )
        coverage_report(view, matches, outlets)
        print(f"removed ambiguous shared-node matches: {len(dropped)}")
        print(f"wrote {match_path(view)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true",
                        help="only rebuild matches from already-downloaded files")
    args = parser.parse_args()
    try:
        with exclusive_file_lock(GRAPHS_DIR / "fetch_graphs.lock"):
            run(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
