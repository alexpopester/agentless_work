#!/bin/bash

# IF USING LOCAL MODEL
#export PATH_TO_LOCAL_MODEL="/home/alex/agentless/llama-3.1-8b.gguf"
#export PROJECT_FILE_LOC="/home/alex/agentless/repo_structure/repo_structures"
export PROJECT_FILE_LOC="/home/apope/documents/neural-networks/swebenchlite/SWE-bench_Lite/repo_structure/repo_structures"

in_dir="og_results"
out_dir="."
out_file="solo_semantic_scores.jsonl"


# USE THIS to run one instance
#python -m agentless.fl.semantic_scoring \
#    --output_file "$out_file" \
#    --output_folder "$out_dir" \
#   --target_id=astropy__astropy-6938 \
#    --input_file "$in_dir/file_level_combined/combined_locs.jsonl"

# USE THIS to run all the instances
#python -m agentless.fl.semantic_scoring \
#    --output_file "$out_file" \
#    --output_folder "$out_dir" \
#    --input_file "$in_dir/file_level_combined/combined_locs.jsonl"