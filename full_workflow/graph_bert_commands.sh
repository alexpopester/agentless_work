#!/bin/bash

#python -m agentless.retrieval.retrieve_context \
python -m agentless.fl.semantic_scoring \
    --output_folder "$1/file_level_combined" \
    --input_file "$1/file_level_combined/combined_locs.jsonl" \
    #--target_id=astropy__astropy-6938 \
    #--repo_path "/path/to/benchmarks/repositories"