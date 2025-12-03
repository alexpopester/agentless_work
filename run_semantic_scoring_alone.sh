#!/bin/bash

# IF using APIS
export GOOGLE_API_KEY="<TODO>"

# IF USING LOCAL MODEL
#export PATH_TO_LOCAL_MODEL="/home/alex/agentless/llama-3.1-8b.gguf"
#export PROJECT_FILE_LOC="/home/alex/agentless/repo_structure/repo_structures"
export PROJECT_FILE_LOC="/home/apope/documents/neural-networks/swebenchlite/SWE-bench_Lite/repo_structure/repo_structures"

in_dir="og_results"
out_dir="."
out_file="solo_semantic_scores.jsonl"

python -m agentless.fl.semantic_scoring \
    --output_file "$out_file" \
    --output_folder "$out_dir" \
    --input_file "$in_dir/file_level_combined/combined_locs.jsonl"