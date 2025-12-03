import json
import numpy as np
from datasets import load_dataset
from unidiff import PatchSet
from tqdm import tqdm
import io

def get_gold_standard_locations(patch_text):
    """
    Parses a git patch to find exactly which files and lines were modified.
    Returns: Dict { 'filename': set(list_of_changed_line_numbers) }
    """
    ground_truth = {}
    try:
        patch = PatchSet(io.StringIO(patch_text))
        for patched_file in patch:
            if patched_file.is_binary_file:
                continue
            
            # Normalize path (remove a/ b/ prefixes)
            path = patched_file.path
            
            changed_lines = []
            for hunk in patched_file:
                start = hunk.source_start
                length = hunk.source_length
                changed_lines.extend(range(start, start + length))
            
            # Only add if there are actual line changes
            if changed_lines:
                ground_truth[path] = set(changed_lines)
                
    except Exception:
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

    patch_lines = ground_truth[f_path]
    func_lines = set(range(f_start, f_end + 1))

    # Returns True if any line in the function overlaps with the patch
    return not func_lines.isdisjoint(patch_lines)

def calculate_average_precision(ranked_items, ground_truth):
    """
    Calculates Average Precision (AP) for a single instance.
    Used to compute MAP across the whole dataset.
    """
    hits = 0
    sum_precisions = 0
    
    # We define total relevant items as the number of modified files 
    # (A proxy, since we don't know exactly how many functions are buggy without AST parsing)
    total_relevant = len(ground_truth.keys())
    if total_relevant == 0:
        return 0

    for i, item in enumerate(ranked_items):
        rank = i + 1
        if is_hit(item, ground_truth):
            hits += 1
            precision_at_i = hits / rank
            sum_precisions += precision_at_i
            
    if hits == 0:
        return 0
    
    return sum_precisions / total_relevant

def calculate_metrics(ranked_file_path):
    print("Loading SWE-bench dataset (for Ground Truth)...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    gold_map = {item["instance_id"]: item["patch"] for item in dataset}

    print(f"Loading Results from {ranked_file_path}...")
    with open(ranked_file_path, "r") as f:
        results = [json.loads(line) for line in f]

    # --- Metric Storage ---
    # Hit Rate: Did we find at least one bug?
    hit_rates = {1: [], 3: [], 5: []}
    
    # Precision: (Relevant Retrieved) / k
    precisions_item = {1: [], 3: [], 5: []}
    
    # Recall: (Relevant Retrieved) / (Total Relevant in Ground Truth)
    recalls_item = {1: [], 3: [], 5: []}
    
    # F1: 2 * (P*R)/(P+R)
    f1_scores = {1: [], 3: [], 5: []}

    # Context Precision: Line-level efficiency
    context_precisions = []
    
    # Ranking Quality
    mrr_list = []
    map_list = []

    for entry in tqdm(results, desc="Evaluating"):
        inst_id = entry["instance_id"]
        if inst_id not in gold_map:
            continue

        patch_text = gold_map[inst_id]
        ground_truth = get_gold_standard_locations(patch_text)
        
        # If ground truth is empty (e.g., binary files only), skip
        if not ground_truth:
            continue

        ranked_items = entry.get("semantic_scoring_results", [])
        
        # --- 1. Calculate MAP (Mean Average Precision) ---
        # We calculate this on the full list (or top 10/20 if capped)
        ap = calculate_average_precision(ranked_items, ground_truth)
        map_list.append(ap)

        # --- 2. Calculate MRR (Mean Reciprocal Rank) ---
        first_hit_rank = None
        for i, item in enumerate(ranked_items):
            if is_hit(item, ground_truth):
                first_hit_rank = i + 1
                break
        
        mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        mrr_list.append(mrr)

        # --- 3. Calculate Cut-based Metrics (@1, @3, @5) ---
        # Total distinct files modified in the patch (Proxy for Total Relevant)
        total_relevant_ground_truth = len(ground_truth.keys())

        for k in [1, 3, 5]:
            # Slice the top k items
            top_k_items = ranked_items[:k]
            
            # Count relevant items in top k
            relevant_retrieved = sum(1 for item in top_k_items if is_hit(item, ground_truth))
            
            # A. Hit Rate (Accuracy) - Did we find *any* bug?
            hit_rates[k].append(1 if relevant_retrieved > 0 else 0)
            
            # B. Precision @ k
            # Denominator is k (the number of items we suggested)
            # If we suggest fewer than k items, use actual length
            denom = len(top_k_items) if len(top_k_items) > 0 else 1
            p_k = relevant_retrieved / denom
            precisions_item[k].append(p_k)
            
            # C. Recall @ k
            # Denominator is Total Relevant Files
            r_k = relevant_retrieved / total_relevant_ground_truth if total_relevant_ground_truth > 0 else 0
            # Cap recall at 1.0 (in case multiple functions map to 1 file)
            r_k = min(r_k, 1.0)
            recalls_item[k].append(r_k)
            
            # D. F1 @ k
            if (p_k + r_k) > 0:
                f1 = 2 * (p_k * r_k) / (p_k + r_k)
            else:
                f1 = 0
            f1_scores[k].append(f1)

        # --- 4. Context Precision (Line Level for Top 5) ---
        total_lines = 0
        relevant_lines = 0
        for item in ranked_items[:5]:
            # Estimate lines based on start/end tags
            n_lines = item.get("end_line", 0) - item.get("start_line", 0) + 1
            total_lines += n_lines
            if is_hit(item, ground_truth):
                relevant_lines += n_lines
        
        if total_lines > 0:
            context_precisions.append(relevant_lines / total_lines)
        else:
            context_precisions.append(0)

    # --- PRINT FINAL REPORT ---
    print("\n" + "="*50)
    print(f"   DETAILED SEMANTIC EVALUATION REPORT")
    print("="*50)
    print(f"Total Instances Evaluated: {len(mrr_list)}")
    print("-" * 50)
    
    print("RANKING METRICS:")
    print(f"  MAP (Mean Avg Precision):  {np.mean(map_list):.4f}")
    print(f"  MRR (Mean Reciprocal Rank):{np.mean(mrr_list):.4f}")
    
    print("-" * 50)
    print("HIT RATE (Success Rate - Found at least 1 bug):")
    print(f"  Hit@1: {np.mean(hit_rates[1]):.2%}")
    print(f"  Hit@3: {np.mean(hit_rates[3]):.2%}")
    print(f"  Hit@5: {np.mean(hit_rates[5]):.2%}")

    print("-" * 50)
    print("PRECISION (Item Level - % of suggestions that are bugs):")
    print(f"  Precision@1: {np.mean(precisions_item[1]):.2%}")
    print(f"  Precision@3: {np.mean(precisions_item[3]):.2%}")
    print(f"  Precision@5: {np.mean(precisions_item[5]):.2%}")

    print("-" * 50)
    print("RECALL (Item Level - % of bugs found):")
    print(f"  Recall@1: {np.mean(recalls_item[1]):.2%}")
    print(f"  Recall@3: {np.mean(recalls_item[3]):.2%}")
    print(f"  Recall@5: {np.mean(recalls_item[5]):.2%}")

    print("-" * 50)
    print("F1 SCORE (Harmonic Mean of Precision & Recall):")
    print(f"  F1@1: {np.mean(f1_scores[1]):.4f}")
    print(f"  F1@3: {np.mean(f1_scores[3]):.4f}")
    print(f"  F1@5: {np.mean(f1_scores[5]):.4f}")

    print("-" * 50)
    print("EFFICIENCY:")
    print(f"  Avg Context Precision (Top 5): {np.mean(context_precisions):.2%}")
    print("="*50)

if __name__ == "__main__":
    calculate_metrics("full_results_with_semantic_scoring/swe-bench-lite/file_level_combined/semantic_scoring_combined_locs.jsonl")