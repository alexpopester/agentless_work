import json
import numpy as np
from datasets import load_dataset
from unidiff import PatchSet
import io


def get_gold_standard_lines(patch_text):
    """
    Parses a git patch to find exactly which files and lines were modified.
    Returns: Set of (filename, line_number) tuples.
    """
    ground_truth = set()
    try:
        patch = PatchSet(io.StringIO(patch_text))
        for patched_file in patch:
            if patched_file.is_binary_file:
                continue
            path = patched_file.path
            for hunk in patched_file:
                start = hunk.source_start
                length = hunk.source_length
                for i in range(start, start + max(length, 1)):
                    ground_truth.add((path, i))
    except Exception:
        pass
    return ground_truth


def parse_prediction_entry(entry):
    """
    Parses the final step of the agentless output.
    Returns: List of (filename, set(line_numbers)) representing ranked suggestions.
    """
    found_edit_locs = entry.get("found_edit_locs", [])
    if not found_edit_locs:
        return []

    # Get the last step (final prediction)
    final_step = (
        found_edit_locs[-1] if isinstance(found_edit_locs, list) else found_edit_locs
    )

    # Handle the specific dict structure of Agentless output
    if isinstance(final_step, dict) and len(final_step) == 1:
        # It often looks like {"file_content": {"file.py": ["lines..."]}}
        # or just direct dict depending on version.
        # Based on your previous script, it seemed to be a list of dicts or similar.
        # Let's trust the structure: final_step is the dict containing file keys.
        pass

    # If the entry was a list of steps, the loop in your original script implied
    # taking the first item of a dict? Let's adapt robustly.
    # Structure: [{"file.py": ["loc1", "loc2"]}]

    ranked_predictions = []

    # Iterate through files in the dictionary (usually 1 dict in the list)
    # Note: Using list structure from previous script logic
    target_dict = final_step
    if isinstance(final_step, list):
        target_dict = final_step[0]

    for filename, loc_strings in target_dict.items():
        for loc_str in loc_strings:
            lines = set()
            for part in loc_str.split("\n"):
                part = part.strip()
                if part.startswith("line:"):
                    try:
                        lines.add(int(part.split(":")[1].strip()))
                    except ValueError:
                        pass

            if lines:
                ranked_predictions.append((filename, lines))

    return ranked_predictions


def calculate_metrics(output_file_path):
    print("Loading SWE-bench dataset (for Ground Truth)...")
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        gold_map = {item["instance_id"]: item["patch"] for item in dataset}
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Loading Predictions from {output_file_path}...")
    results = []
    with open(output_file_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # --- Storage for Metric Aggregation ---
    # Ranking Metrics
    ap_scores = []  # Average Precision (for MAP)
    rr_scores = []  # Reciprocal Rank (for MRR)

    # Cutoff Metrics @ K
    k_cuts = [1, 3, 5]
    metrics_at_k = {k: {"hit": [], "prec": [], "rec": [], "f1": []} for k in k_cuts}

    processed_count = 0

    for entry in results:
        inst_id = entry["instance_id"]
        if inst_id not in gold_map:
            continue

        processed_count += 1

        # 1. Ground Truth
        gold_lines = get_gold_standard_lines(gold_map[inst_id])
        if not gold_lines:
            continue  # Skip invalid gold standards

        # 2. Predictions
        preds = parse_prediction_entry(entry)

        # --- CALCULATE PER-INSTANCE METRICS ---

        # A. Determine Relevance of each Ranked Prediction
        # A prediction is "relevant" if it overlaps with ANY gold line
        relevance_vec = []
        for pred_file, pred_lines in preds:
            pred_set = {(pred_file, line) for line in pred_lines}
            is_relevant = not pred_set.isdisjoint(gold_lines)
            relevance_vec.append(1 if is_relevant else 0)

        # B. Reciprocal Rank (MRR)
        # 1 / (rank of first relevant item)
        try:
            first_correct_idx = relevance_vec.index(1)
            rr = 1.0 / (first_correct_idx + 1)
        except ValueError:
            rr = 0.0
        rr_scores.append(rr)

        # C. Average Precision (MAP)
        # Sum(Precision@k * rel(k)) / Total_Relevant_In_List
        # Note: Standard MAP divides by Total_Relevant_In_GT, but for localization lists
        # where we might not find everything, dividing by Total_Relevant_Found is common
        # or Total_Relevant_Possible. Let's use Total_Relevant_Found to punish false positives strictly.
        # Actually, standard IR definition: Sum(P@k)/Num_Relevant_Documents.
        num_relevant = sum(relevance_vec)
        if num_relevant > 0:
            p_sum = 0
            cum_rel = 0
            for i, rel in enumerate(relevance_vec):
                if rel:
                    cum_rel += 1
                    p_sum += cum_rel / (i + 1)
            ap = p_sum / num_relevant
        else:
            ap = 0.0
        ap_scores.append(ap)

        # D. Metrics @ K
        for k in k_cuts:
            # Slice top k
            top_k_preds = preds[:k]
            top_k_rel = relevance_vec[:k]

            # Hit Rate: Did we find at least one?
            is_hit = 1 if sum(top_k_rel) > 0 else 0
            metrics_at_k[k]["hit"].append(is_hit)

            # Precision: % of suggestions that were relevant
            # If k > len(preds), we divide by k (punishing short lists) or len?
            # Standard IR: divide by k.
            prec = sum(top_k_rel) / k
            metrics_at_k[k]["prec"].append(prec)

            # Recall: % of Gold Lines covered by Top K
            # Gather all lines predicted in top k
            covered_lines = set()
            for i in range(min(k, len(top_k_preds))):
                p_file, p_lines = top_k_preds[i]
                for line in p_lines:
                    covered_lines.add((p_file, line))

            overlap = len(covered_lines.intersection(gold_lines))
            rec = overlap / len(gold_lines) if len(gold_lines) > 0 else 0
            metrics_at_k[k]["rec"].append(rec)

            # F1
            if (prec + rec) > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
            else:
                f1 = 0.0
            metrics_at_k[k]["f1"].append(f1)

    # --- FINAL REPORT GENERATION ---
    print("\n" + "=" * 50)
    print("    DETAILED SEMANTIC EVALUATION REPORT")
    print("=" * 50)
    print(f"Total Instances Evaluated: {processed_count}")
    print("-" * 50)

    print("RANKING METRICS:")
    print(f"  MAP (Mean Avg Precision):  {np.mean(ap_scores):.4f}")
    print(f"  MRR (Mean Reciprocal Rank):{np.mean(rr_scores):.4f}")
    print("-" * 50)

    print("HIT RATE (Success Rate - Found at least 1 bug):")
    for k in k_cuts:
        print(f"  Hit@{k}: {np.mean(metrics_at_k[k]['hit']):.2%}")
    print("-" * 50)

    print("PRECISION (Item Level - % of suggestions that are bugs):")
    for k in k_cuts:
        print(f"  Precision@{k}: {np.mean(metrics_at_k[k]['prec']):.2%}")
    print("-" * 50)

    print("RECALL (Item Level - % of bugs found):")
    for k in k_cuts:
        print(f"  Recall@{k}: {np.mean(metrics_at_k[k]['rec']):.2%}")
    print("-" * 50)

    print("F1 SCORE (Harmonic Mean of Precision & Recall):")
    for k in k_cuts:
        print(f"  F1@{k}: {np.mean(metrics_at_k[k]['f1']):.4f}")
    print("-" * 50)

    print("EFFICIENCY:")
    # "Avg Context Precision" is essentially Mean Precision@5
    print(f"  Avg Context Precision (Top 5): {np.mean(metrics_at_k[5]['prec']):.2%}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    # You can hardcode the path here or use argparse
    # FILE_PATH = "/home/apope/documents/neural-networks/agentless_release/agentless_swebench_lite/edit_location_samples/loc_outputs.jsonl"
    FILE_PATH = "/home/apope/documents/neural-networks/Agentless/full_results_without_semantic_scoring/swe-bench-lite/edit_location_samples/loc_outputs.jsonl"

    try:
        calculate_metrics(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
