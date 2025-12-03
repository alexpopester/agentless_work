import json
import torch
import numpy as np

from datasets import load_dataset
from tqdm import tqdm
from threading import Lock

from agentless.util.preprocess_data import (
    check_contains_valid_loc,
    filter_none_python,
    filter_out_test_files,
    get_repo_structure,
    get_repo_files,
)
from agentless.util.utils import (
    load_existing_instance_ids,
    load_jsonl,
    setup_logger,
    write_jsonl,
)

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
import ast


# --- 1. The Code Parser (The "Chopper") ---
class CodeParser(ast.NodeVisitor):
    """
    Parses a python file and extracts all functions and methods
    with their line numbers and source code.
    """

    def __init__(self, source_code, file_path):
        self.source_code = source_code
        self.file_path = file_path
        self.functions = []
        self.lines = source_code.splitlines()

    def visit_FunctionDef(self, node):
        # Extract the function source code specifically
        # (This is a simple extraction, sophisticated versions handle decorators)
        start_line = node.lineno - 1
        end_line = node.end_lineno
        func_source = "\n".join(self.lines[start_line:end_line])

        self.functions.append(
            {
                "name": node.name,
                "file": self.file_path,
                "start_line": start_line + 1,
                "end_line": end_line,
                "code": func_source,
            }
        )
        self.generic_visit(node)


def extract_functions(file_path, source_code):
    try:
        tree = ast.parse(source_code)
        parser = CodeParser(source_code, file_path)
        parser.visit(tree)
        return parser.functions
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


# --- 2. The AI Engine (GraphCodeBERT) ---
class SemanticMatcher:
    def __init__(self):
        print("Loading GraphCodeBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        # 3. Load the Model
        self.model = AutoModel.from_pretrained("fine-tune-graphcodebert-bugs")
        #self.model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text):
        # Truncate to 512 because BERT cannot handle infinite text
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # return outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # [CLS] token is usually the first token (index 0) representing the whole sequence
        # --- FIX: Mean Pooling ---
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Expand mask to match embedding dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Sum embeddings and divide by valid token count
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return (sum_embeddings / sum_mask).cpu().numpy()


# --- 3. Putting it together (The Workflow) ---
def run_analysis(bug_report_text, file_contents_map):
    matcher = SemanticMatcher()

    # A. Embed the Bug Report
    print("Embedding Bug Report...")
    bug_vec = matcher.get_embedding(bug_report_text)

    all_candidates = []

    # B. Process every file Agentless found
    for file_path, source_code in file_contents_map.items():
        # 1. Chop file into functions
        functions = extract_functions(file_path, source_code)

        for func in functions:
            # 2. Embed the function source code
            code_vec = matcher.get_embedding(func["code"])

            # 3. Compare (Cosine Similarity)
            score = cosine_similarity(bug_vec, code_vec)[0][0]

            all_candidates.append(
                {
                    "file": file_path,
                    "function_name": func["name"],
                    "score": float(score),
                    "start_line": func["start_line"],
                    "end_line": func["end_line"],
                }
            )

    # C. Sort by relevance
    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    return all_candidates


# --- Configuration ---
MODEL_NAME = "microsoft/graphcodebert-base"


def localize_repo(bug, args, swe_bench_data, existing_instance_ids, write_lock=None):
    instance_id = bug["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    # logger = setup_logger(log_file)
    # logger.info(f"Processing bug {instance_id}")

    # if bug["instance_id"] in existing_instance_ids:
    #    logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
    #    return

    # logger.info(f"================ localize {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = (
        bench_data["problem_statement"] + "\n" + bench_data["hints_text"]
    )
    structure = get_repo_structure(
        instance_id, bug["repo"], bug["base_commit"], "playground"
    )
    filter_none_python(structure)  # some basic filtering steps
    filter_out_test_files(structure)
    start_file_locs = load_jsonl(args.input_file)
    for entry in start_file_locs:
        if entry["instance_id"] == instance_id:
            found_files_list = entry["found_files"]
            break
    files_map = get_repo_files(structure, found_files_list)
    results = run_analysis(problem_statement, files_map)

    entry["semantic_scoring_results"] = results

    print(f"\nTop Suspect Functions for Bug:")
    for res in results:
        print(
            f"[{res['score']:.4f}] {res['function_name']} (File: {res['file']}, Line {res['start_line']})"
        )

    return entry


def produce_semantic_analysis(args):
    swe_bench_data = load_dataset(args.dataset, split="test")
    #        existing_instance_ids = (
    #            load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    #        )
    existing_instance_ids = set()
    # if args.num_threads == 1:
    new_entry_list = []
    count = 0
    for bug in tqdm(swe_bench_data, colour="MAGENTA"):
        result = localize_repo(bug, args, swe_bench_data, existing_instance_ids)
        if result:
            new_entry_list.append(result)
        count += 1
        if count % 10 == 0:
            write_jsonl(
                new_entry_list, os.path.join(args.output_folder, args.output_file)
            )

    write_jsonl(new_entry_list, os.path.join(args.output_folder, args.output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", required=True, help="Path to localize.py output (jsonl)"
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--output_file", type=str, default="semantic_scoring_combined_locs.jsonl"
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="Path to save the ranked output",
    )
    # parser.add_argument("--repo_path", required=True, help="Root path of the project repositories")
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    args = parser.parse_args()
    print(args)

    produce_semantic_analysis(args)
