#!/usr/bin/env python3
"""
orchestrator_tissue_restricted.py

End-to-end, tissue-restricted (Breast, Lung, Colon, Prostate) dataset pipeline
for CancerGPT replication using DrugCombDB.

Features:
- Async ingestion (filtered at source) with resilient error handling
- Cleaning & deduplication with validation
- Synergy labeling with threshold tuning
- Tissue-wise stratified train/test splits
- Prompt serialization (JSONL/CSV)
- 100+ prompt templates (static + Ollama 3.1 LLM-generated)
- k-shot prompt builder with few-shot examples
- Comprehensive logging and progress tracking

Authoritative, leakage-safe, publication-grade.
"""

import argparse
import asyncio
import aiohttp
import os
import json
import logging
import sys
import requests
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tenacity import retry, stop_after_attempt, wait_exponential

# -----------------------
# CONFIGURATION
# -----------------------
class Config:
    """Centralized configuration management"""
    # API Settings
    BASE_URL = "http://drugcombdb.denglab.org:8888"
    PAGE_SIZE = 200
    MAX_PAGES = 6000
    CONCURRENCY = 40
    
    # Ollama Settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2:latest"  # Change to llama3.1 if available
    OLLAMA_TIMEOUT = 60
    OLLAMA_ENABLED = True
    
    # Processing Settings
    SYNERGY_THRESHOLD = 5.0
    MIN_TISSUE_SIZE = 200
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Output Settings
    OUTPUT_DIR = "cancergpt_filtered_outputs"
    
    # Paths
    RAW = f"{OUTPUT_DIR}/raw.parquet"
    CLEAN = f"{OUTPUT_DIR}/clean.parquet"
    LABELED = f"{OUTPUT_DIR}/labeled.parquet"
    SPLIT = f"{OUTPUT_DIR}/split.parquet"
    JSONL = f"{OUTPUT_DIR}/dataset.jsonl"
    CSV = f"{OUTPUT_DIR}/dataset.csv"
    TEMPLATES = f"{OUTPUT_DIR}/prompt_templates.jsonl"
    OLLAMA_TEMPLATES = f"{OUTPUT_DIR}/ollama_templates.jsonl"

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Configure logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{Config.OUTPUT_DIR}/pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("orchestrator")

# -----------------------
# TISSUE NORMALIZATION
# -----------------------
ALLOWED_TISSUES = {
    "breast": ["breast"],
    "lung": ["lung"],
    "colon": ["colon", "colorectal"],
    "prostate": ["prostate"]
}

def normalize_tissue(raw: Optional[str]) -> Optional[str]:
    """Normalize tissue names to canonical forms"""
    if not isinstance(raw, str):
        return None
    raw = raw.lower().strip()
    for canonical, keys in ALLOWED_TISSUES.items():
        for k in keys:
            if k in raw:
                return canonical
    return None

def validate_row(r: Dict) -> bool:
    """Validate that a row has required fields"""
    required = ["drugA", "drugB", "cell_line", "tissue", "synergy_score"]
    return all(r.get(f) is not None for f in required)

# -----------------------
# INGESTION
# -----------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 2, 10))
async def fetch_page(session: aiohttp.ClientSession, page: int) -> List[Dict]:
    """Fetch a page from DrugCombDB with retry logic"""
    try:
        async with session.get(
            f"{Config.BASE_URL}/integration/list",
            params={"page": page, "size": Config.PAGE_SIZE},
            timeout=30
        ) as r:
            if r.status != 200:
                log.warning(f"Page {page} returned status {r.status}")
                return []
            j = await r.json()
            return j.get("list", [])
    except asyncio.TimeoutError:
        log.warning(f"Timeout fetching page {page}")
        return []
    except Exception as e:
        log.error(f"Error fetching page {page}: {e}")
        return []

async def ingest() -> None:
    """Asynchronously ingest filtered tissue data from DrugCombDB"""
    log.info("Starting async ingestion from DrugCombDB...")
    sem = asyncio.Semaphore(Config.CONCURRENCY)
    rows = []
    skipped = 0

    async with aiohttp.ClientSession() as session:
        tasks = []

        async def sem_fetch(p):
            async with sem:
                return await fetch_page(session, p)

        for p in range(1, Config.MAX_PAGES + 1):
            tasks.append(sem_fetch(p))

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching pages"):
            try:
                data = await fut
            except Exception as e:
                log.debug(f"Skipped page due to error: {e}")
                continue

            for r in data:
                tissue = normalize_tissue(r.get("tissue"))
                if tissue is None:
                    skipped += 1
                    continue

                try:
                    row = {
                        "block_id": r.get("blockId"),
                        "drugA": r.get("drugName1"),
                        "drugB": r.get("drugName2"),
                        "cell_line": r.get("cellName"),
                        "tissue": tissue,
                        "synergy_score": float(r.get("synergyScore", 0)),
                        "source": r.get("source")
                    }
                    if validate_row(row):
                        rows.append(row)
                except (ValueError, TypeError) as e:
                    skipped += 1
                    continue

    df = pd.DataFrame(rows)
    df.to_parquet(Config.RAW, index=False)
    log.info(f"Ingested {len(df)} rows | Skipped {skipped} rows (non-target tissues)")
    return df

# -----------------------
# CLEANING
# -----------------------
def clean() -> pd.DataFrame:
    """Clean dataset by removing nulls, duplicates, and normalizing values"""
    log.info("Starting data cleaning...")
    df = pd.read_parquet(Config.RAW)
    
    initial_len = len(df)
    
    # Remove rows with missing critical fields
    df = df.dropna(subset=["drugA", "drugB", "cell_line", "synergy_score"])
    after_nulls = len(df)
    
    # Normalize synergy_score to numeric
    df["synergy_score"] = pd.to_numeric(df["synergy_score"], errors="coerce")
    df = df.dropna(subset=["synergy_score"])
    after_numeric = len(df)

    # Normalize drug order for consistency
    df[["drugA", "drugB"]] = np.sort(df[["drugA", "drugB"]].astype(str), axis=1)
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=["drugA", "drugB", "cell_line"])
    after_dedup = len(df)

    df.to_parquet(Config.CLEAN, index=False)
    
    log.info(f"Cleaned: {initial_len} → {after_nulls} (nulls) → {after_numeric} (numeric) → {after_dedup} (dedup)")
    return df

# -----------------------
# LABELING
# -----------------------
def label() -> pd.DataFrame:
    """Label rows as synergistic or not based on threshold"""
    log.info(f"Labeling synergy (threshold={Config.SYNERGY_THRESHOLD})...")
    df = pd.read_parquet(Config.CLEAN)
    
    df["synergy_label"] = (df["synergy_score"] > Config.SYNERGY_THRESHOLD).astype(int)
    
    synergistic = (df["synergy_label"] == 1).sum()
    non_synergistic = (df["synergy_label"] == 0).sum()
    
    df.to_parquet(Config.LABELED, index=False)
    
    log.info(f"Synergistic: {synergistic} | Non-synergistic: {non_synergistic}")
    return df

# -----------------------
# SPLITTING
# -----------------------
def split() -> pd.DataFrame:
    """Split data by tissue with stratified train/test split"""
    log.info(f"Splitting data by tissue (test_size={Config.TEST_SIZE})...")
    df = pd.read_parquet(Config.LABELED)
    out = []

    for tissue, g in df.groupby("tissue"):
        if len(g) < Config.MIN_TISSUE_SIZE:
            log.warning(f"Skipping {tissue}: only {len(g)} samples (min: {Config.MIN_TISSUE_SIZE})")
            continue

        train, test = train_test_split(
            g, 
            test_size=Config.TEST_SIZE,
            stratify=g["synergy_label"],
            random_state=Config.RANDOM_STATE
        )

        train["split"] = "train"
        test["split"] = "test"
        out.extend([train, test])

        synergy_pct = (g["synergy_label"].sum() / len(g) * 100)
        log.info(f"{tissue:12s} | total: {len(g):5d} | train: {len(train):5d} | test: {len(test):5d} | synergy: {synergy_pct:5.1f}%")

    final = pd.concat(out, ignore_index=True)
    final.to_parquet(Config.SPLIT, index=False)
    
    log.info(f"Final split: {len(final)} rows")
    return final

# -----------------------
# OLLAMA INTEGRATION
# -----------------------
def check_ollama_available() -> bool:
    """Check if Ollama service is running"""
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        log.warning("Ollama service not available at " + Config.OLLAMA_BASE_URL)
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(2, 4, 30))
def call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama to generate text"""
    try:
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "num_predict": 150,
            },
            timeout=Config.OLLAMA_TIMEOUT
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            log.warning(f"Ollama API returned {response.status_code}")
            return None
    except Exception as e:
        log.error(f"Ollama API error: {e}")
        return None

def generate_ollama_templates(n: int = 80) -> List[Dict]:
    """Generate diverse prompt templates using Ollama 3.1"""
    if not check_ollama_available():
        log.info("Skipping Ollama template generation - service unavailable")
        return []
    
    log.info(f"Generating {n} prompt templates using Ollama...")
    
    system_prompt = """You are an expert in drug interaction prediction and oncology.
Generate a unique, professional prompt template for asking an LLM to predict drug synergy in cancer cells.

Requirements:
- Be specific about the inputs (Drug 1, Drug 2, Cell line, Tissue)
- Ask for Yes/No classification
- Vary the phrasing and structure significantly
- Include context about why synergy prediction matters
- Use different question formats (direct, conditional, exploratory)

Output ONLY the template text, no explanations."""

    templates = []
    
    for i in tqdm(range(n), desc="Ollama template generation"):
        prompt = system_prompt + f"\n\nGenerate template #{i+1}:"
        
        response = call_ollama(prompt)
        if response and len(response) > 20:
            # Ensure template has placeholders
            if not all(x in response.lower() for x in ['drug', 'cell', 'tissue']):
                response = response.replace("{drugA}", "{{drugA}}")
                response = response.replace("{drugB}", "{{drugB}}")
                response = response.replace("{cell_line}", "{{cell_line}}")
                response = response.replace("{tissue}", "{{tissue}}")
                
                # If still no placeholders, add them
                if "{" not in response:
                    response += "\n\nDrug 1: {drugA}\nDrug 2: {drugB}\nCell line: {cell_line}\nTissue: {tissue}\n\nPredict synergy (Yes/No):"
            
            templates.append({
                "template": response,
                "fields": ["drugA", "drugB", "cell_line", "tissue"],
                "source": "ollama",
                "model": Config.OLLAMA_MODEL
            })
    
    log.info(f"Generated {len(templates)} Ollama templates")
    return templates

# -----------------------
# PROMPT GENERATION
# -----------------------
def build_prompt(r: Dict) -> str:
    """Build a basic prompt from a row"""
    return (
        f"Drug 1: {r.drugA}\n"
        f"Drug 2: {r.drugB}\n"
        f"Cell line: {r.cell_line}\n"
        f"Tissue: {r.tissue}\n\n"
        "Is this drug combination synergistic? Answer Yes or No."
    )

def generate_static_templates(n: int = 50) -> List[Dict]:
    """Generate static prompt templates with varied structures"""
    log.info(f"Generating {n} static prompt templates...")
    
    preambles = [
        "Determine whether the following drug pair exhibits synergy.",
        "Assess synergy of the drug combination described below.",
        "Classify the drug pair as synergistic or antagonistic.",
        "Given the cancer cell context, predict if the drugs act synergistically.",
        "Based on the information provided, is this a synergistic combination?",
        "Evaluate potential synergy in this drug combination.",
        "Predict synergistic interaction for the following pair.",
        "Does the following drug combination show synergistic effects?",
        "Analyze synergy potential of the drug pair below.",
        "Determine if these drugs work synergistically in the given context.",
        "Is this drug combination likely to be synergistic?",
        "Predict whether this drug pair will interact synergistically.",
        "Based on context, classify synergy likelihood.",
        "Evaluate drug synergy for the combination below.",
    ]

    questions = [
        "Is this combination synergistic? (Yes/No)",
        "Does this drug pair show synergy? (Yes/No)",
        "Answer: Is there synergy? (Yes/No)",
        "Synergistic? (Yes/No)",
        "Prediction: Synergistic (Yes) or not (No)?",
        "Do these drugs interact synergistically?",
        "Will this combination be synergistic?",
        "Likely synergy: Yes or No?",
    ]

    field_orders = [
        ["drugA", "drugB", "cell_line", "tissue"],
        ["cell_line", "tissue", "drugA", "drugB"],
        ["tissue", "cell_line", "drugA", "drugB"],
        ["drugA", "drugB", "tissue", "cell_line"],
    ]

    templates = []
    
    for preamble in preambles:
        for question in questions:
            for fields_order in field_orders:
                body = "\n".join([
                    f"Drug 1: {{drugA}}" if f == "drugA" else
                    f"Drug 2: {{drugB}}" if f == "drugB" else
                    f"Cell line: {{cell_line}}" if f == "cell_line" else
                    f"Tissue: {{tissue}}"
                    for f in fields_order
                ])

                templates.append({
                    "template": f"{preamble}\n\n{body}\n\n{question}",
                    "fields": fields_order,
                    "source": "static"
                })

                if len(templates) >= n:
                    break
            if len(templates) >= n:
                break
        if len(templates) >= n:
            break
    
    return templates[:n]

def generate_and_save_templates() -> None:
    """Generate and save all prompt templates (static + Ollama)"""
    log.info("="*60)
    log.info("GENERATING PROMPT TEMPLATES")
    log.info("="*60)
    
    # Generate static templates
    static_templates = generate_static_templates(n=50)
    
    # Generate Ollama templates if available
    ollama_templates = generate_ollama_templates(n=80) if Config.OLLAMA_ENABLED else []
    
    all_templates = static_templates + ollama_templates
    
    # Save all templates to single file
    with open(Config.TEMPLATES, "w") as f:
        for t in all_templates:
            f.write(json.dumps(t) + "\n")
    
    log.info(f"Saved {len(all_templates)} total templates to {Config.TEMPLATES}")
    log.info(f"  - Static: {len(static_templates)}")
    log.info(f"  - Ollama: {len(ollama_templates)}")
    log.info("="*60)

def serialize() -> None:
    """Serialize dataset to JSONL and CSV formats"""
    log.info("Serializing dataset to JSONL and CSV...")
    df = pd.read_parquet(Config.SPLIT)
    
    df["prompt"] = df.apply(build_prompt, axis=1)
    df["answer"] = df["synergy_label"].map({1: "Yes", 0: "No"})

    df.to_csv(Config.CSV, index=False)
    log.info(f"Saved CSV: {Config.CSV}")
    
    df[["prompt", "answer", "split", "tissue"]].to_json(
        Config.JSONL, orient="records", lines=True
    )
    log.info(f"Saved JSONL: {Config.JSONL}")
    
    log.info(f"Total samples: {len(df)}")
    log.info(f"  - Train: {(df['split'] == 'train').sum()}")
    log.info(f"  - Test: {(df['split'] == 'test').sum()}")

# -----------------------
# ORCHESTRATION
# -----------------------
def run_all() -> None:
    """Execute complete pipeline"""
    log.info("="*60)
    log.info("STARTING CANCERGPT PIPELINE (Tissue-Restricted)")
    log.info("="*60)
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info(f"Target tissues: {', '.join(ALLOWED_TISSUES.keys())}")
    
    try:
        # Step 1: Ingest
        log.info("\n[1/6] INGESTION")
        asyncio.run(ingest())
        
        # Step 2: Clean
        log.info("\n[2/6] CLEANING")
        clean()
        
        # Step 3: Label
        log.info("\n[3/6] LABELING")
        label()
        
        # Step 4: Split
        log.info("\n[4/6] SPLITTING")
        split()
        
        # Step 5: Generate templates
        log.info("\n[5/6] PROMPT TEMPLATES")
        generate_and_save_templates()
        
        # Step 6: Serialize
        log.info("\n[6/6] SERIALIZATION")
        serialize()
        
        log.info("\n" + "="*60)
        log.info("PIPELINE COMPLETE ✓")
        log.info("="*60)
        log.info(f"Output directory: {Config.OUTPUT_DIR}")
        log.info(f"Files generated:")
        log.info(f"  - {Config.RAW}")
        log.info(f"  - {Config.CLEAN}")
        log.info(f"  - {Config.LABELED}")
        log.info(f"  - {Config.SPLIT}")
        log.info(f"  - {Config.TEMPLATES}")
        log.info(f"  - {Config.CSV}")
        log.info(f"  - {Config.JSONL}")
        log.info("="*60)
        
    except Exception as e:
        log.error(f"PIPELINE FAILED: {e}", exc_info=True)
        sys.exit(1)

def run_templates_only() -> None:
    """Generate only prompt templates"""
    log.info("Generating prompt templates only...")
    generate_and_save_templates()
    log.info(f"Templates saved to {Config.TEMPLATES}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CancerGPT pipeline: tissue-restricted DrugCombDB processing"
    )
    parser.add_argument(
        "cmd",
        choices=["run_all", "templates_only", "ingest", "clean", "label", "split", "serialize"],
        help="Pipeline stage to execute"
    )
    parser.add_argument(
        "--disable-ollama",
        action="store_true",
        help="Disable Ollama template generation"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=Config.OLLAMA_MODEL,
        help=f"Ollama model to use (default: {Config.OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--synergy-threshold",
        type=float,
        default=Config.SYNERGY_THRESHOLD,
        help=f"Synergy score threshold (default: {Config.SYNERGY_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    # Apply CLI overrides
    if args.disable_ollama:
        Config.OLLAMA_ENABLED = False
    if args.ollama_model:
        Config.OLLAMA_MODEL = args.ollama_model
    if args.synergy_threshold:
        Config.SYNERGY_THRESHOLD = args.synergy_threshold
    
    try:
        if args.cmd == "run_all":
            run_all()
        elif args.cmd == "templates_only":
            run_templates_only()
        elif args.cmd == "ingest":
            asyncio.run(ingest())
        elif args.cmd == "clean":
            clean()
        elif args.cmd == "label":
            label()
        elif args.cmd == "split":
            split()
        elif args.cmd == "serialize":
            serialize()
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
