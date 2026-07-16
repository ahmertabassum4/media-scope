"""Regenerate the paper's LLM similarity graph for our outlet universe.

Paper-guided procedure from §3.1 "LLM-Graph": for each domain, prompt the model
with the published prompt for 5 similar websites wrapped in <s>...</s> tags,
convert responses to JSONL, build the deduplicated level-0 graph, and expand
recursively to level 3 (nodes discovered at level 3 are leaves and are never
queried — matching their node counts).

The API key is read from the OPENAI_API_KEY environment variable only. With no
key, --dry-run still exercises the full pipeline (prompt building, response
parsing, caching, graph assembly) against canned responses so the code path is
verified before spending money.

Resumable: every response is appended to data/graphs/llm_responses.jsonl and
already-queried domains are skipped on restart.

Outputs: data/graphs/llm_edges.csv (source,target ids), llm_name2id.json,
and graph_match_llm.json (outlet key -> node id).

Run (user):
  export OPENAI_API_KEY=sk-...
  nohup /opt/anaconda3/bin/python3 baselines/multiview/llm_graph.py \
      > data/graphs/llm_graph.log 2>&1 &
Track:  wc -l data/graphs/llm_responses.jsonl; tail data/graphs/llm_graph.log
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    GRAPHS_DIR,
    atomic_write_text,
    candidate_exact_hosts,
    deduplicate_node_matches,
    exclusive_file_lock,
    load_mbfc_sources,
    load_outlets,
    match_path,
    mbfc_name_index,
    normalized_host,
    outlet_universe_sha256,
    sha256_file,
)

RESPONSES_JSONL = GRAPHS_DIR / "llm_responses.jsonl"
DRYRUN_RESPONSES_JSONL = GRAPHS_DIR / "llm_responses_dryrun_v2.jsonl"
LLM_EDGES = GRAPHS_DIR / "llm_edges.csv"
LLM_NAME2ID = GRAPHS_DIR / "llm_name2id.json"

# Verbatim prompt from the paper (§3.1, LLM-Graph).
PROMPT_TEMPLATE = (
    "Based on similarity, give me 5 similar websites to, {domain}. "
    "Return only the websites URL, strictly without any explanation, "
    "don't add numbers in start. Each website should be wrapped under the tag <s>"
)
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
# Paper used gpt-3.5-turbo-0125; pass --model gpt-4o-mini if it is retired.
DEFAULT_MODEL = "gpt-3.5-turbo-0125"
MAX_LEVEL = 3  # query levels 0..2; level-3 nodes are leaves (paper's expansion)

TAG_RE = re.compile(r"<s>\s*(.*?)\s*</s>", re.IGNORECASE | re.DOTALL)


def parse_similar(response_text, self_domain=None):
    """Registered domains from a '<s>url</s>' response, deduplicated in order."""
    seen, out = set(), []
    for raw in TAG_RE.findall(str(response_text or "")):
        dom = normalized_host(raw)
        if dom and dom != self_domain and dom not in seen:
            seen.add(dom)
            out.append(dom)
    return out[:5]


def load_cache(path=RESPONSES_JSONL, expected_model=None,
               expected_prompt_sha256=None):
    """domain -> list of similar domains, from previous runs."""
    cache = {}
    if not Path(path).exists():
        return cache
    with open(path) as fh:
        lines = fh.readlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    final_nonempty = nonempty[-1] if nonempty else -1
    for line_index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if line_index == final_nonempty:
                print(f"warning: ignoring interrupted final cache line in {path}")
                continue
            raise ValueError(
                f"{path}: malformed JSON at line {line_index + 1}"
            ) from None
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: cache line {line_index + 1} is not an object")
        domain = normalized_host(rec.get("domain"))
        if not domain:
            raise ValueError(f"{path}: invalid cached domain")
        model = rec.get("model")
        if expected_model and model != expected_model:
            raise ValueError(
                f"{path}: cached model {model!r} does not match {expected_model!r}"
            )
        prompt_sha256 = rec.get("prompt_sha256")
        if expected_prompt_sha256 and prompt_sha256 != expected_prompt_sha256:
            raise ValueError(
                f"{path}: cached prompt hash {prompt_sha256!r} does not match "
                f"{expected_prompt_sha256!r}"
            )
        raw_similar = rec.get("similar", [])
        if not isinstance(raw_similar, list) or len(raw_similar) > 5:
            raise ValueError(
                f"{path}: invalid similar-site list at line {line_index + 1}"
            )
        similar = []
        for value in raw_similar:
            host = normalized_host(value)
            if host and host != domain and host not in similar:
                similar.append(host)
        if domain in cache and cache[domain] != similar:
            raise ValueError(f"{path}: conflicting cached responses for {domain}")
        cache[domain] = similar
    return cache


def append_response(record, path=RESPONSES_JSONL):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell():
            fh.seek(fh.tell() - 1)
            if fh.read(1) != "\n":
                fh.write("\n")
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class OpenAIQuerier:
    """One prompt -> one raw response string, with basic retry/backoff."""

    def __init__(self, model, max_retries=5):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Export it in the environment "
                "(or use --dry-run to verify the pipeline without a key)."
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def __call__(self, domain):
        prompt = PROMPT_TEMPLATE.format(domain=domain)
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - API/network errors
                if "model" in str(exc).lower() and "not" in str(exc).lower():
                    raise SystemExit(
                        f"model {self.model!r} unavailable: {exc}\n"
                        "Re-run with --model gpt-4o-mini (documented deviation)."
                    ) from exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")


class DryRunQuerier:
    """Deterministic canned responses exercising the exact parse format."""

    CANNED = ["cnn.com", "theguardian.com", "aljazeera.com", "nytimes.com", "reuters.com"]

    def __init__(self, model="dry-run"):
        self.model = model

    def __call__(self, domain):
        picks = [d for d in self.CANNED if d != domain][:5]
        return "\n".join(f"<s> https://www.{d}/ </s>" for d in picks)


def expand_graph(seeds, querier, cache, max_level=MAX_LEVEL, max_queries=0,
                 responses_path=RESPONSES_JSONL, log_every=100):
    """Level-by-level expansion; returns levels, edges, and completeness."""
    level_of = {}
    frontier = sorted(set(seeds))
    for dom in frontier:
        level_of[dom] = 0
    edges = set()
    n_queries = 0
    complete = True
    for level in range(max_level):  # only levels 0..max_level-1 get queried
        next_frontier = []
        for dom in frontier:
            if dom in cache:
                similar = cache[dom]
            else:
                if max_queries and n_queries >= max_queries:
                    print(f"stopping: --max-queries {max_queries} reached at level {level}")
                    complete = False
                    break
                text = querier(dom)
                similar = parse_similar(text, self_domain=dom)
                append_response({"domain": dom, "level": level, "model": querier.model,
                                 "prompt_sha256": PROMPT_SHA256,
                                 "response": text, "similar": similar},
                                path=responses_path)
                cache[dom] = similar
                n_queries += 1
                if n_queries % log_every == 0:
                    print(f"level {level}: {n_queries} new queries, "
                          f"{len(level_of)} nodes so far", flush=True)
            for sim in similar:
                edges.add((dom, sim))
                if sim not in level_of:
                    level_of[sim] = level + 1
                    next_frontier.append(sim)
        if not complete:
            break
        frontier = sorted(set(next_frontier))
        print(f"level {level} done: {len(level_of)} nodes, {len(edges)} directed edges, "
              f"frontier for level {level + 1}: {len(frontier)}", flush=True)
    return level_of, edges, complete


def build_outputs(level_of, edges, outlets, mbfc_by_name, model, max_level,
                  responses_path, suffix="", complete=True, seed_count=None):
    edges_path = LLM_EDGES.with_name(f"llm_edges{suffix}.csv")
    name2id_path = LLM_NAME2ID.with_name(f"llm_name2id{suffix}.json")
    matches_path = match_path("llm").with_name(f"graph_match_llm{suffix}.json")
    domains = sorted(level_of)
    if not domains:
        raise ValueError("cannot write an empty LLM graph")
    name2id = {dom: i for i, dom in enumerate(domains)}
    unknown_edge_nodes = sorted({node for edge in edges for node in edge} - set(name2id))
    if unknown_edge_nodes:
        raise ValueError(f"LLM graph edges contain unknown nodes: {unknown_edge_nodes[:10]}")
    atomic_write_text(name2id_path, json.dumps(name2id, indent=0, sort_keys=True))
    edge_buffer = io.StringIO(newline="")
    writer = csv.writer(edge_buffer)
    writer.writerow(["source", "target"])
    for src, dst in sorted(edges):
        writer.writerow([name2id[src], name2id[dst]])
    atomic_write_text(edges_path, edge_buffer.getvalue())
    matches = {}
    for key in sorted(outlets):
        for dom in candidate_exact_hosts(outlets[key], mbfc_by_name):
            if dom in name2id:
                matches[key] = name2id[dom]
                break
    matches, dropped = deduplicate_node_matches(matches, outlets)
    atomic_write_text(matches_path, json.dumps(matches, indent=0, sort_keys=True))
    manifest_path = GRAPHS_DIR / f"llm_graph_manifest{suffix}.json"
    manifest = {
        "schema": "multiview_llm_graph_v1",
        "complete": bool(complete),
        "expansion_complete": True,
        "model": model,
        "max_level": max_level,
        "prompt_sha256": PROMPT_SHA256,
        "seed_domains": seed_count,
        "outlet_universe_sha256": outlet_universe_sha256(outlets),
        "nodes": len(domains),
        "directed_edges": len(edges),
        "matched_outlets": len(matches),
        "dropped_shared_node_matches": dropped,
        "outputs": {
            "edges": {"path": str(edges_path.resolve()), "sha256": sha256_file(edges_path)},
            "name2id": {"path": str(name2id_path.resolve()), "sha256": sha256_file(name2id_path)},
            "matches": {"path": str(matches_path.resolve()), "sha256": sha256_file(matches_path)},
            "responses": {
                "path": str(Path(responses_path).resolve()),
                "sha256": sha256_file(responses_path),
            },
        },
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"LLM graph: {len(domains)} nodes, {len(edges)} directed edges")
    print(f"matched outlets: {len(matches)}/{len(outlets)}")
    print(f"removed ambiguous shared-node matches: {len(dropped)}")
    print(f"wrote {edges_path}, {name2id_path}, {matches_path}, {manifest_path}")


def seed_domains(outlets, mbfc_by_name):
    """One canonical seed host per outlet; duplicate outlet domains merge."""
    seeds = set()
    for outlet in outlets.values():
        candidates = candidate_exact_hosts(outlet, mbfc_by_name)
        if candidates:
            seeds.add(candidates[0])
    return sorted(seeds)


def output_suffix(dry_run, limit_seeds, max_level):
    parts = []
    if dry_run:
        parts.append("dryrun")
    if limit_seeds:
        parts.append(f"seeds{limit_seeds}")
    if max_level != MAX_LEVEL:
        parts.append(f"level{max_level}")
    return "_" + "_".join(parts) if parts else ""


def run(args):
    outlets = load_outlets()
    mbfc_by_name = mbfc_name_index(load_mbfc_sources())
    seeds = seed_domains(outlets, mbfc_by_name)
    if args.limit_seeds:
        seeds = seeds[:args.limit_seeds]
    if not seeds:
        raise SystemExit("no valid seed domains found")
    effective_model = "dry-run" if args.dry_run else args.model
    print(f"seeds (level 0): {len(seeds)} domains; model={effective_model}; "
          f"max_level={args.max_level}; dry_run={args.dry_run}")

    responses_path = DRYRUN_RESPONSES_JSONL if args.dry_run else RESPONSES_JSONL
    querier = DryRunQuerier() if args.dry_run else OpenAIQuerier(args.model)
    cache = load_cache(
        responses_path, expected_model=querier.model,
        expected_prompt_sha256=PROMPT_SHA256,
    )
    print(f"cached responses: {len(cache)}")
    level_of, edges, complete = expand_graph(
        seeds, querier, cache, max_level=args.max_level,
        max_queries=args.max_queries, responses_path=responses_path,
    )
    if not complete:
        print("response cache updated; graph outputs were not written because expansion is incomplete")
        return
    suffix = output_suffix(args.dry_run, args.limit_seeds, args.max_level)
    paper_complete = (
        not args.dry_run and not args.limit_seeds and args.max_level == MAX_LEVEL
    )
    build_outputs(
        level_of, edges, outlets, mbfc_by_name, querier.model,
        args.max_level, responses_path, suffix=suffix,
        complete=paper_complete, seed_count=len(seeds),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-level", type=int, default=MAX_LEVEL,
                        help="expansion depth (paper: 3; levels 0..2 are queried)")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="safety cap on NEW API queries this run (0 = unlimited)")
    parser.add_argument("--limit-seeds", type=int, default=0,
                        help="cap the number of seed outlets (smoke tests)")
    parser.add_argument("--dry-run", action="store_true",
                        help="no API key needed: canned responses, full pipeline")
    args = parser.parse_args()
    if args.max_level < 1:
        parser.error("--max-level must be at least 1")
    if args.max_queries < 0:
        parser.error("--max-queries cannot be negative")
    if args.limit_seeds < 0:
        parser.error("--limit-seeds cannot be negative")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_file_lock(GRAPHS_DIR / "llm_graph.lock"):
            run(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
