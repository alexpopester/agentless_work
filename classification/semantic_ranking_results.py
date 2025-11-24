import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from unidiff import PatchSet
from tqdm import tqdm
import io


def get_gold_standard_locations(patch_text):
    """
    Parses a git patch to find exactly which files and lines were modified.
    Returns: Dict { 'filename': [list_of_changed_line_numbers] }
    """
    ground_truth = {}
    try:
        patch = PatchSet(io.StringIO(patch_text))
        for patched_file in patch:
            # Skip binary files or deleted files if needed
            if patched_file.is_binary_file:
                continue

            # Get the path (removing a/ and b/ prefixes usually found in git diffs)
            path = patched_file.path

            changed_lines = []
            for hunk in patched_file:
                # We care about the lines in the SOURCE file (before the fix)
                # because Agentless is looking at the buggy code.
                start = hunk.source_start
                length = hunk.source_length
                # Add all lines covered by this hunk
                changed_lines.extend(range(start, start + length))

            ground_truth[path] = set(changed_lines)
    except Exception as e:
        # Some patches might be malformed or complex
        pass
    return ground_truth


def is_hit(ranked_func, ground_truth):
    """
    Checks if a specific ranked function overlaps with the ground truth patch.
    """
    f_path = ranked_func.get("file")
    f_start = ranked_func.get("start_line", 0)
    f_end = ranked_func.get("end_line", 0)

    if f_path not in ground_truth:
        return False

    # Check for overlap: Did the patch modify lines INSIDE this function?
    # Intersection of (Function Range) and (Patch Range)
    patch_lines = ground_truth[f_path]
    func_lines = set(range(f_start, f_end + 1))

    return not func_lines.isdisjoint(patch_lines)


def calculate_metrics(ranked_file_path):
    print("Loading SWE-bench dataset (for Ground Truth)...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    # Create map for fast lookup
    gold_map = {item["instance_id"]: item["patch"] for item in dataset}

    print(f"Loading Results from {ranked_file_path}...")
    with open(ranked_file_path, "r") as f:
        results = [json.loads(line) for line in f]

    # Storage for metrics
    recalls = {1: [], 3: [], 5: []}
    reciprocal_ranks = []
    precisions = []

    for entry in tqdm(results, desc="Evaluating"):
        inst_id = entry["instance_id"]
        if inst_id not in gold_map:
            continue

        # 1. Get Ground Truth
        patch_text = gold_map[inst_id]
        ground_truth = get_gold_standard_locations(patch_text)

        # 2. Get Your Rankings
        # Assuming your structure is entry['suspicious_functions'] or similar
        # If you used the previous script, it might be entry['semantic_rankings']
        ranked_items = entry.get("semantic_scoring_results", [])

        # If no items found, scores are 0
        if not ranked_items:
            recalls[1].append(0)
            recalls[3].append(0)
            recalls[5].append(0)
            reciprocal_ranks.append(0)
            precisions.append(0)
            continue

        # 3. Calculate MRR & Recall
        first_hit_rank = None
        hits_in_top_5 = 0

        for i, item in enumerate(ranked_items[:5]):  # Only check top 5
            rank = i + 1
            if is_hit(item, ground_truth):
                if first_hit_rank is None:
                    first_hit_rank = rank
                hits_in_top_5 += 1

        # Recall logic
        recalls[1].append(1 if first_hit_rank == 1 else 0)
        recalls[3].append(1 if first_hit_rank and first_hit_rank <= 3 else 0)
        recalls[5].append(1 if first_hit_rank and first_hit_rank <= 5 else 0)

        # MRR logic
        mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        reciprocal_ranks.append(mrr)

        # 4. Calculate Context Precision (for Top 5)
        # Precision = (Size of Relevant Code / Size of Total Retrieved Code)
        # We estimate size using line counts
        total_lines_retrieved = 0
        relevant_lines_retrieved = 0

        for item in ranked_items[:5]:
            # Calculate length of this snippet
            # If you saved 'snippet', count \n. Or use lines list length.
            # Assuming item['lines'] is a list of lines or lines range
            if "lines" in item and isinstance(item["lines"], list):
                n_lines = len(item["lines"])
            else:
                # Fallback estimate
                n_lines = item.get("end_line", 0) - item.get("start_line", 0)

            total_lines_retrieved += n_lines

            if is_hit(item, ground_truth):
                relevant_lines_retrieved += n_lines

        if total_lines_retrieved > 0:
            precisions.append(relevant_lines_retrieved / total_lines_retrieved)
        else:
            precisions.append(0)

    # --- Print Report ---
    print("\n" + "=" * 40)
    print("   SEMANTIC FUNNEL PERFORMANCE REPORT   ")
    print("=" * 40)
    print(f"Total Instances Evaluated: {len(results)}")
    print(f"Recall@1: {np.mean(recalls[1]):.2%}")
    print(f"Recall@3: {np.mean(recalls[3]):.2%}")
    print(f"Recall@5: {np.mean(recalls[5]):.2%}")
    print("-" * 20)
    print(f"Mean Reciprocal Rank (MRR): {np.mean(reciprocal_ranks):.4f}")
    print("-" * 20)
    print(f"Avg Context Precision (Top 5): {np.mean(precisions):.2%}")
    print("=" * 40)


if __name__ == "__main__":
    # Point this to your actual output file
    calculate_metrics(
        "full_results_with_semantic_scoring/swe-bench-lite/file_level_combined/semantic_scoring_combined_locs.jsonl"
    )
