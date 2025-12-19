#!/usr/bin/env python3
"""
orchestrator.py

Production-grade orchestrator to:
 - Ingest DrugCombDB integration list (async, batched)
 - Clean, deduplicate, label (synergy > 5 => label=1)
 - Tissue-wise splits (train/test)
 - Build LLM prompts and write JSONL / CSV
 - Generate 100+ diverse prompt templates and save templates JSONL
 - Provide a helper to assemble k-shot prompts using chosen templates

Usage:
    pip install aiohttp pandas numpy tqdm scikit-learn pyarrow aiofiles tenacity
    python orchestrator.py --help

Examples:
    python orchestrator.py run_all
    python orchestrator.py gen_templates --count 200
    python orchestrator.py build_kshot --k 5 --template-index 12 --out sample_kshot.txt
"""

import argparse
import asyncio
import aiohttp
import aiofiles
import os
import json
import logging
import math
import time
from typing import List, Dict, Any, Iterable
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -----------------------
# CONFIG
# -----------------------
BASE_URL = "http://drugcombdb.denglab.org:8888"
PAGE_SIZE = 200
MAX_PAGES = 6000         # set high to attempt full DB; you can lower for testing
CONCURRENCY = 40
OUTPUT_DIR = "data_pipeline_outputs"
RAW_PARQUET = os.path.join(OUTPUT_DIR, "raw_integration.parquet")
CLEAN_PARQUET = os.path.join(OUTPUT_DIR, "clean_dataset.parquet")
LABELED_PARQUET = os.path.join(OUTPUT_DIR, "labeled_dataset.parquet")
SPLIT_PARQUET = os.path.join(OUTPUT_DIR, "final_split_dataset.parquet")
JSONL_PROMPTS = os.path.join(OUTPUT_DIR, "cancergpt_dataset.jsonl")
CSV_FULL = os.path.join(OUTPUT_DIR, "cancergpt_dataset.csv")
TEMPLATE_JSONL = os.path.join(OUTPUT_DIR, "prompt_templates.jsonl")
LOGFILE = os.path.join(OUTPUT_DIR, "orchestrator.log")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "orchestrator.checkpoint.json")
FLUSH_BATCH = 50000  # flush rows to disk every N rows

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("orchestrator")

# -----------------------
# UTIL: Checkpointing
# -----------------------
def read_checkpoint() -> Dict[str, Any]:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def write_checkpoint(state: Dict[str, Any]):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

# -----------------------
# INGEST: Async fetch pages
# -----------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception))
async def fetch_page(session: aiohttp.ClientSession, page: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/integration/list"
    params = {"page": page, "size": PAGE_SIZE}
    async with session.get(url, params=params, timeout=30) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"Bad status {resp.status} for page {page}: {text[:200]}")
        j = await resp.json()
        # API returns a dict with "list" key in the earlier HTML example
        if isinstance(j, dict) and "list" in j:
            return j.get("list", [])
        # fallback if the API gives a list directly
        if isinstance(j, list):
            return j
        # otherwise return empty
        return []

async def ingest_pages(start_page: int = 1, end_page: int = MAX_PAGES) -> None:
    """
    Ingest pages asynchronously and stream results to Parquet in chunks.
    """
    logger.info("Starting ingestion pages %d..%d", start_page, end_page)
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=60)

    async def sem_fetch(p):
        async with sem:
            try:
                return await fetch_page(session, p)
            except Exception as e:
                logger.error("Failed page %d: %s", p, e)
                return []

    rows_buffer = []
    total_rows = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        pages = list(range(start_page, end_page + 1))
        # We'll use asyncio.as_completed so we can flush incrementally
        tasks = [asyncio.create_task(sem_fetch(p)) for p in pages]

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="fetching pages"):
            try:
                entries = await fut
            except Exception as e:
                logger.exception("Error awaiting task: %s", e)
                entries = []
            for r in entries:
                # normalize fields (based on the API frontend)
                rows_buffer.append({
                    "block_id": r.get("blockId"),
                    "drugA": r.get("drugName1"),
                    "drugB": r.get("drugName2"),
                    "cell_line": r.get("cellName"),
                    "tissue": r.get("tissue"),
                    "synergy_score": r.get("synergyScore"),
                    "source": r.get("source")
                })
            total_rows += len(entries)

            # flush periodically
            if len(rows_buffer) >= FLUSH_BATCH:
                df = pd.DataFrame(rows_buffer)
                if os.path.exists(RAW_PARQUET):
                    df.to_parquet(RAW_PARQUET, compression="snappy", engine="pyarrow", index=False, append=True)
                else:
                    df.to_parquet(RAW_PARQUET, compression="snappy", engine="pyarrow", index=False)
                logger.info("Flushed %d rows to %s (total so far ~%d)", len(rows_buffer), RAW_PARQUET, total_rows)
                rows_buffer.clear()

    # final flush
    if rows_buffer:
        df = pd.DataFrame(rows_buffer)
        if os.path.exists(RAW_PARQUET):
            # append to existing parquet by reading existing and concat (pyarrow append to parquet files isn't native in pandas)
            df.to_parquet(RAW_PARQUET, compression="snappy", engine="pyarrow", index=False)
        else:
            df.to_parquet(RAW_PARQUET, compression="snappy", engine="pyarrow", index=False)
        logger.info("Final flush %d rows", len(rows_buffer))

    logger.info("Finished ingestion; estimated total rows: %d", total_rows)

def run_ingest(start_page: int = 1, end_page: int = MAX_PAGES):
    """
    wrapper to run the async ingestion from sync context.
    """
    asyncio.run(ingest_pages(start_page=start_page, end_page=end_page))

# -----------------------
# CLEANING
# -----------------------
def clean_dataset():
    logger.info("Starting cleaning step")
    if not os.path.exists(RAW_PARQUET):
        raise FileNotFoundError("Raw parquet not found. Run ingest first.")
    df = pd.read_parquet(RAW_PARQUET)
    logger.info("Raw rows: %d", len(df))

    # Drop rows with missing core fields
    df = df.dropna(subset=["drugA", "drugB", "cell_line", "synergy_score"])
    # Convert score to numeric
    df["synergy_score"] = pd.to_numeric(df["synergy_score"], errors="coerce")
    df = df.dropna(subset=["synergy_score"])
    # canonical order drug names so pair (A,B) == (B,A)
    df[["drugA", "drugB"]] = np.sort(df[["drugA", "drugB"]].astype(str), axis=1)
    # deduplicate pair x cell_line
    before = len(df)
    df = df.drop_duplicates(subset=["drugA", "drugB", "cell_line"])
    logger.info("Dropped %d duplicates", before - len(df))

    df.to_parquet(CLEAN_PARQUET, compression="snappy", index=False)
    logger.info("Saved cleaned parquet: %s (%d rows)", CLEAN_PARQUET, len(df))

# -----------------------
# LABELING
# -----------------------
SYNERGY_THRESHOLD = 5.0

def label_synergy(threshold: float = SYNERGY_THRESHOLD):
    logger.info("Starting labeling with threshold %s", threshold)
    if not os.path.exists(CLEAN_PARQUET):
        raise FileNotFoundError("Clean parquet not found. Run clean first.")
    df = pd.read_parquet(CLEAN_PARQUET)
    df["synergy_label"] = (df["synergy_score"].astype(float) > float(threshold)).astype(int)
    df.to_parquet(LABELED_PARQUET, compression="snappy", index=False)
    logger.info("Saved labeled parquet: %s (%d rows)", LABELED_PARQUET, len(df))

# -----------------------
# SPLITTING BY TISSUE
# -----------------------
MIN_TISSUE_SIZE = 200  # only split tissues with at least this many rows

def split_by_tissue(min_size: int = MIN_TISSUE_SIZE):
    logger.info("Starting tissue-wise split (min_size=%d)", min_size)
    if not os.path.exists(LABELED_PARQUET):
        raise FileNotFoundError("Labeled parquet not found. Run labeling first.")
    df = pd.read_parquet(LABELED_PARQUET)
    out_parts = []
    keep_count = 0
    for tissue, g in df.groupby("tissue"):
        if len(g) < min_size:
            continue
        keep_count += len(g)
        train, test = train_test_split(g, test_size=0.2, stratify=g["synergy_label"], random_state=42)
        train["split"] = "train"
        test["split"] = "test"
        out_parts.append(train)
        out_parts.append(test)
        logger.info("Tissue %s -> train %d, test %d", tissue, len(train), len(test))

    if not out_parts:
        raise RuntimeError("No tissues met min_size requirement. Lower min_size.")
    final_df = pd.concat(out_parts, ignore_index=True)
    final_df.to_parquet(SPLIT_PARQUET, compression="snappy", index=False)
    logger.info("Saved split parquet: %s (%d rows kept across tissues)", SPLIT_PARQUET, keep_count)

# -----------------------
# PROMPT BUILDING
# -----------------------
def build_prompt_text(row: pd.Series, template: str = None) -> str:
    """
    Build the basic default prompt or use a template with placeholders:
      {drugA}, {drugB}, {cell_line}, {tissue}, {synergy_score}
    """
    if template is None:
        return (
            f"Drug 1: {row.drugA}\n"
            f"Drug 2: {row.drugB}\n"
            f"Cell line: {row.cell_line}\n"
            f"Tissue: {row.tissue}\n\n"
            "Is this drug combination synergistic? Answer Yes or No."
        )
    else:
        # template placeholders are the same names as column keys
        return template.format(
            drugA=row.drugA,
            drugB=row.drugB,
            cell_line=row.cell_line,
            tissue=row.tissue,
            synergy_score=row.synergy_score
        )

def serialize_prompts(template_index: int = None):
    logger.info("Starting prompt serialization")
    if not os.path.exists(SPLIT_PARQUET):
        raise FileNotFoundError("Split parquet not found. Run split_by_tissue first.")
    df = pd.read_parquet(SPLIT_PARQUET)
    # load templates if requested
    chosen_template = None
    templates = []
    if template_index is not None:
        if not os.path.exists(TEMPLATE_JSONL):
            raise FileNotFoundError("Template file does not exist. Generate templates first.")
        with open(TEMPLATE_JSONL, "r", encoding="utf-8") as f:
            templates = [json.loads(line) for line in f]
        if template_index < 0 or template_index >= len(templates):
            raise IndexError("template_index out of range")
        chosen_template = templates[template_index]["template"]

    # create prompt & answer and write JSONL and CSV
    with open(JSONL_PROMPTS, "w", encoding="utf-8") as jf:
        rows = []
        for _, r in df.iterrows():
            prompt = build_prompt_text(r, template=chosen_template)
            answer = "Yes" if int(r.synergy_label) == 1 else "No"
            rec = {
                "block_id": r.block_id,
                "drugA": r.drugA,
                "drugB": r.drugB,
                "cell_line": r.cell_line,
                "tissue": r.tissue,
                "synergy_score": r.synergy_score,
                "prompt": prompt,
                "answer": answer,
                "split": r.split
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
        # also write CSV
        pd.DataFrame(rows).to_csv(CSV_FULL, index=False)
    logger.info("Serialized prompts to %s (CSV: %s)", JSONL_PROMPTS, CSV_FULL)

# -----------------------
# PROMPT TEMPLATE GENERATOR (>=100 templates)
# -----------------------
def generate_prompt_templates(min_count: int = 100, out_path: str = TEMPLATE_JSONL):
    """
    Programmatically generate diverse prompt templates by combining:
      - various instruction lead-ins
      - varying order of fields
      - optional extra context fields (synergy_score placeholder)
      - answer formats (Yes/No, Likert, probability)
    Saves templates to JSONL (one template per line with metadata).
    """
    logger.info("Generating prompt templates, target >= %d", min_count)

    intro_phrases = [
        "Assess whether the following drug pair is synergistic.",
        "Decide whether the combination below demonstrates drug synergy.",
        "Given the experimental context, label the combination as synergistic or not.",
        "From the data below, determine if the two drugs act synergistically.",
        "Read the cell line information and state whether the two drugs are synergistic.",
        "Determine if the drug combination shows synergistic effect in the specified cell line.",
        "Judge whether the following drug pair is synergistic (Yes / No).",
        "Analyze the pair of drugs and the cell line — is there synergy?",
        "Based on the provided sample, answer if the combination exhibits synergy.",
        "Indicate whether the combination of the two drugs is synergistic."
    ]

    field_orders = [
        ["drugA", "drugB", "cell_line", "tissue"],
        ["drugB", "drugA", "tissue", "cell_line"],
        ["cell_line", "tissue", "drugA", "drugB"],
        ["drugA", "cell_line", "drugB", "tissue"],
    ]

    question_phrases = [
        "Is this drug combination synergistic? Answer Yes or No.",
        "Synergy? (Yes / No)",
        "Would you label this pair as synergistic? Reply 'Yes' or 'No'.",
        "Does this combination show synergy? Provide 'Yes' or 'No'.",
        "Return 'Yes' if synergistic, otherwise 'No'.",
    ]

    extra_includes = [
        "",  # no extra
        "Provide a short reason (one sentence) after the label.",
        "Answer with 'Yes' or 'No' followed by the likely numeric synergy score range (low/medium/high).",
        "Also provide a confidence score between 0 and 1 after your label.",
    ]

    answer_formats = [
        "{label}\n",  # default
        "{label} — {reason}",  # label + reason
        "Label: {label}",       # labelled format
    ]

    templates = []
    # Build templates combinatorially until min_count reached
    for intro in intro_phrases:
        for order in field_orders:
            for q in question_phrases:
                for extra in extra_includes:
                    # Build field block
                    fields_text_lines = []
                    for fld in order:
                        if fld == "drugA":
                            fields_text_lines.append("Drug 1: {drugA}")
                        elif fld == "drugB":
                            fields_text_lines.append("Drug 2: {drugB}")
                        elif fld == "cell_line":
                            fields_text_lines.append("Cell line: {cell_line}")
                        elif fld == "tissue":
                            fields_text_lines.append("Tissue: {tissue}")
                    # Optional include synergy_score prompt
                    # create two variants: with or without synergy_score placeholder
                    for include_score in (False, True):
                        body = "\n".join(fields_text_lines)
                        if include_score:
                            body += "\nSynergy score (if known): {synergy_score}"
                        # join into a template string
                        extra_str = ("\n" + extra) if extra else ""
                        template_text = f"{intro}\n\n{body}\n\n{q}{extra_str}"
                        templates.append({
                            "template": template_text,
                            "intro": intro,
                            "order": order,
                            "question": q,
                            "extra": extra,
                            "include_score": include_score
                        })
                    if len(templates) >= min_count:
                        break
                if len(templates) >= min_count:
                    break
            if len(templates) >= min_count:
                break
        if len(templates) >= min_count:
            break

    # If combinatorics didn't reach min_count for some reason, augment by permutations
    i = 0
    while len(templates) < min_count:
        t = templates[i % len(templates)].copy()
        t["template"] += f"\n# variant {len(templates)+1}"
        templates.append(t)
        i += 1
        if i > 10000:
            break

    # Save templates to JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for t in templates:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    logger.info("Wrote %d templates to %s", len(templates), out_path)
    return templates

# -----------------------
# HELPER: Build k-shot prompt
# -----------------------
def make_k_shot_prompt(template_text: str, examples: Iterable[Dict[str, Any]], query: Dict[str, Any],
                       prefix: str = None, suffix: str = None) -> str:
    """
    Compose a k-shot prompt using:
      - template_text (template with placeholders)
      - examples: iterable of dicts with keys: drugA, drugB, cell_line, tissue, synergy_score, answer (Yes/No)
      - query: single example dict (without answer)
      - optional prefix/suffix
    Returns assembled prompt string (few-shot).
    """
    parts = []
    if prefix:
        parts.append(prefix.strip())

    # Put examples
    for ex in examples:
        ex_prompt = template_text.format(
            drugA=ex["drugA"],
            drugB=ex["drugB"],
            cell_line=ex["cell_line"],
            tissue=ex.get("tissue", ""),
            synergy_score=ex.get("synergy_score", "")
        )
        parts.append(ex_prompt.strip())
        # include the answer line in a canonical format
        parts.append(f"Answer: {ex['answer']}")

    # Now append the query
    q_prompt = template_text.format(
        drugA=query["drugA"],
        drugB=query["drugB"],
        cell_line=query["cell_line"],
        tissue=query.get("tissue", ""),
        synergy_score=query.get("synergy_score", "")
    )
    parts.append(q_prompt.strip())
    if suffix:
        parts.append(suffix.strip())
    return "\n\n".join(parts)

# -----------------------
# CLI / Orchestration
# -----------------------
def run_all(args):
    # check checkpoint
    ckpt = read_checkpoint()
    step = ckpt.get("last_step", "start")
    logger.info("Starting run_all from step: %s", step)

    if step == "start":
        # ingest
        try:
            run_ingest(start_page=1, end_page=args.max_pages)
            write_checkpoint({"last_step": "ingested"})
            step = "ingested"
        except Exception:
            logger.exception("Ingestion failed")
            raise

    if step == "ingested":
        clean_dataset()
        write_checkpoint({"last_step": "cleaned"})
        step = "cleaned"

    if step == "cleaned":
        label_synergy(threshold=args.threshold)
        write_checkpoint({"last_step": "labeled"})
        step = "labeled"

    if step == "labeled":
        split_by_tissue(min_size=args.min_tissue_size)
        write_checkpoint({"last_step": "split"})
        step = "split"

    if step == "split":
        # generate templates if not exist
        if not os.path.exists(TEMPLATE_JSONL):
            generate_prompt_templates(min_count=args.templates)
        serialize_prompts(template_index=None)
        write_checkpoint({"last_step": "done"})
        step = "done"

    logger.info("Run_all completed. final step: %s", step)

def gen_templates(args):
    generate_prompt_templates(min_count=args.count, out_path=TEMPLATE_JSONL)

def build_kshot_cli(args):
    # loads templates, dataset, then sample examples to build k-shot using template index
    if not os.path.exists(TEMPLATE_JSONL):
        raise FileNotFoundError("Templates not found; run gen_templates first.")
    with open(TEMPLATE_JSONL, "r", encoding="utf-8") as f:
        templates = [json.loads(line) for line in f]
    template = templates[args.template_index]["template"]
    # load dataset
    if not os.path.exists(SPLIT_PARQUET):
        raise FileNotFoundError("Split dataset not found; run run_all first.")
    df = pd.read_parquet(SPLIT_PARQUET)
    # sample k examples from train
    train = df[df["split"] == "train"]
    # ensure we have mix of labels
    pos = train[train["synergy_label"] == 1].sample(n=min(len(train[train["synergy_label"]==1]), args.k), replace=False)
    neg = train[train["synergy_label"] == 0].sample(n=min(len(train[train["synergy_label"]==0]), args.k), replace=False)
    exs = []
    # pick alternating pos/neg examples until k reached
    i = 0
    while len(exs) < args.k:
        if i % 2 == 0 and len(pos) > 0:
            r = pos.iloc[i//2 % len(pos)]
        else:
            r = neg.iloc[i//2 % len(neg)]
        exs.append({
            "drugA": r.drugA, "drugB": r.drugB, "cell_line": r.cell_line,
            "tissue": r.tissue, "synergy_score": r.synergy_score, "answer": "Yes" if int(r.synergy_label) == 1 else "No"
        })
        i += 1
        if i > 10000:
            break
    # pick a random query
    query_row = df.sample(1).iloc[0]
    query = {
        "drugA": query_row.drugA, "drugB": query_row.drugB, "cell_line": query_row.cell_line,
        "tissue": query_row.tissue, "synergy_score": query_row.synergy_score
    }
    prompt = make_k_shot_prompt(template, exs, query, prefix=args.prefix, suffix=args.suffix)
    # write to output file
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(prompt)
    logger.info("Wrote k-shot prompt to %s (k=%d, template_index=%d)", args.out, args.k, args.template_index)


def main():
    p = argparse.ArgumentParser(prog="orchestrator")
    sub = p.add_subparsers(dest="cmd")

    r_all = sub.add_parser("run_all", help="Run the full pipeline sequentially")
    r_all.add_argument("--max-pages", type=int, default=1000, help="Max pages to fetch (limits runtime). Default 1000")
    r_all.add_argument("--threshold", type=float, default=SYNERGY_THRESHOLD, help="Synergy threshold")
    r_all.add_argument("--min-tissue-size", type=int, default=MIN_TISSUE_SIZE)
    r_all.add_argument("--templates", type=int, default=120, help="Number of templates to generate if not present")

    gen = sub.add_parser("gen_templates", help="Generate prompt templates")
    gen.add_argument("--count", type=int, default=150, help="How many templates to create")

    kshot = sub.add_parser("build_kshot", help="Build a k-shot prompt file using existing templates")
    kshot.add_argument("--k", type=int, default=5, help="Number of few-shot examples")
    kshot.add_argument("--template-index", type=int, default=0, help="Index of template to use (0-based)")
    kshot.add_argument("--out", type=str, default="kshot_prompt.txt", help="Output file")
    kshot.add_argument("--prefix", type=str, default=None, help="Optional prefix instruction")
    kshot.add_argument("--suffix", type=str, default=None, help="Optional suffix instruction")

    args = p.parse_args()
    if args.cmd is None:
        p.print_help()
        return

    try:
        if args.cmd == "run_all":
            run_all(args)
        elif args.cmd == "gen_templates":
            gen_templates(args)
        elif args.cmd == "build_kshot":
            build_kshot_cli(args)
        else:
            raise RuntimeError("Unknown command")
    except Exception as e:
        logger.exception("Command failed: %s", e)
        raise

if __name__ == "__main__":
    main()
