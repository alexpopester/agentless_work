import os
import ast
import logging
import subprocess
import argparse
import random
import json  # <--- NEW: For saving logs
from typing import List, Dict, Optional

import torch
import whatthepatch
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

# --- Configuration ---
REPO_DIR = "./playground"
OUTPUT_MODEL_DIR = "./fine-tuned-graphcodebert-bugs"
LOG_FILE = "./training_logs.json"  # <--- NEW: Where to save the graph data
BASE_MODEL = "microsoft/graphcodebert-base"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 1. The Improved Code Parser (With Class Context) ---
class ContextAwareParser(ast.NodeVisitor):
    def __init__(self, source_code, file_path):
        self.source_code = source_code
        self.file_path = file_path
        self.functions = []
        self.lines = source_code.splitlines()
        self.current_class = None

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        start_line = node.lineno - 1
        end_line = node.end_lineno
        func_source = "\n".join(self.lines[start_line:end_line])

        if len(func_source) > 2000:
            func_source = func_source[:2000] + "\n# ... (truncated)"

        qualified_name = (
            f"{self.current_class}.{node.name}" if self.current_class else node.name
        )

        self.functions.append(
            {
                "name": qualified_name,
                "file": self.file_path,
                "start_line": start_line + 1,
                "end_line": end_line,
                "code_to_embed": f"# Function: {qualified_name}\n# File: {self.file_path}\n{func_source}",
            }
        )


def extract_functions(source_code, file_path):
    try:
        tree = ast.parse(source_code)
        parser = ContextAwareParser(source_code, file_path)
        parser.visit(tree)
        return parser.functions
    except Exception:
        return []


# --- 2. Git & File Utils ---
def clone_repo_if_missing(repo_name):
    safe_name = repo_name.replace("/", "__")
    repo_path = os.path.join(REPO_DIR, safe_name)

    if not os.path.exists(repo_path):
        logger.info(f"Cloning {repo_name} to {repo_path}...")
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_name}", repo_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return repo_path


def get_file_content_at_commit(repo_path, commit_hash, file_path):
    try:
        cmd = ["git", "show", f"{commit_hash}:{file_path}"]
        result = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True, errors="ignore"
        )
        if result.returncode == 0:
            return result.stdout
    except Exception as e:
        logger.error(f"Error fetching file {file_path}: {e}")
    return None


# --- 3. Data Processing ---
def generate_dataset(dataset_split, limit=None):
    data_rows = []
    print("Generating Training Triplets...")
    count = 0

    for instance in tqdm(dataset_split):
        if limit and count >= limit:
            break

        repo_name = instance["repo"]
        repo_path = clone_repo_if_missing(repo_name)
        base_commit = instance["base_commit"]
        patch_text = instance["patch"]

        diffs = whatthepatch.parse_patch(patch_text)

        for diff in diffs:
            if not diff.header or not diff.changes:
                continue

            raw_path = diff.header.old_path
            if raw_path.startswith("a/") or raw_path.startswith("b/"):
                file_path = raw_path[2:]
            else:
                file_path = raw_path

            source_code = get_file_content_at_commit(repo_path, base_commit, file_path)
            if not source_code:
                continue

            modified_lines = set()
            for change in diff.changes:
                if change.old is not None:
                    modified_lines.add(change.old)

            if not modified_lines:
                continue

            functions = extract_functions(source_code, file_path)
            buggy_funcs = []
            clean_funcs = []

            for func in functions:
                func_range = set(range(func["start_line"], func["end_line"] + 1))
                if not func_range.isdisjoint(modified_lines):
                    buggy_funcs.append(func["code_to_embed"])
                else:
                    clean_funcs.append(func["code_to_embed"])

            if buggy_funcs and clean_funcs:
                data_rows.append(
                    {
                        "anchor": instance["problem_statement"],
                        "positive": buggy_funcs[0],
                        "negative": random.choice(clean_funcs),
                    }
                )
                count += 1

    print(f"Generated {len(data_rows)} triplets.")
    if not data_rows:
        return None
    return Dataset.from_list(data_rows)


# --- 4. Main Execution ---
def main():
    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR)

    print("Loading SWE-bench train split...")
    processed_dataset_path = "./processed_bug_dataset"

    if os.path.exists(processed_dataset_path):
        print(f"Loading cached dataset from {processed_dataset_path}...")
        train_dataset = load_from_disk(processed_dataset_path)
    else:
        print("No cached dataset found. Generating from scratch...")
        raw_dataset = load_dataset("princeton-nlp/SWE-bench", split="train")
        train_dataset = generate_dataset(raw_dataset)

        if not train_dataset:
            print("ERROR: No triplets generated.")
            return

        print(f"Saving dataset to {processed_dataset_path} for future use...")
        train_dataset.save_to_disk(processed_dataset_path)

    # --- SANITY CHECK: Verify Data Quality ---
    print("\n" + "="*50)
    print("DATASET INSPECTION (Example 0)")
    print("="*50)
    
    example = train_dataset[0]
    
    print(f"[ANCHOR] (The Bug Report):\n{example['anchor'][:300]}...\n") 
    print("-" * 20)
    print(f"[POSITIVE] (The Code matching the bug):\n{example['positive'][:300]}...\n")
    print("-" * 20)
    print(f"[NEGATIVE] (Random unrelated code):\n{example['negative'][:300]}...\n")
    print("="*50 + "\n")

    # Ask for user confirmation before burning GPU time
    confirm = input("Does the data above look correct? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborting training.")
        return

    print(f"Initializing {BASE_MODEL}...")
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 512
    model.tokenizer.model_max_length = 512
    model.tokenizer.truncation = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training on {device}")

    train_loss = losses.Mut(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True if device == "cuda" else False,
        logging_steps=50,       # <--- UPDATED: Log more frequently for smoother graphs
        save_strategy="epoch",
        report_to="none",       # Keep it local
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    print("Starting training...")
    # trainer.train()
    trainer.train(resume_from_checkpoint=False) # Changed to False for fresh run unless you specifically want to resume

    # --- SAVE LOGS FOR GRAPHING ---
    print(f"Saving training logs to {LOG_FILE}...")
    
    # log_history is a list of dicts like: [{'loss': 0.5, 'step': 10, 'epoch': 0.1}, ...]
    history = trainer.state.log_history

    # Filter out entries that don't have 'loss' (some contain only epoch info)
    filtered_history = [entry for entry in history if "loss" in entry]
    
    with open(LOG_FILE, "w") as f:
        json.dump(filtered_history, f, indent=4)

    print(f"Logs saved! You can load {LOG_FILE} to plot Loss vs Step.")
    
    print(f"Saving model to {OUTPUT_MODEL_DIR}...")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    print("Done!")

if __name__ == "__main__":
    main()