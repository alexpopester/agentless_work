#!/bin/bash

# Repair generation
python -m agentless.repair.repair --loc_file $1/edit_location_individual/loc_merged_0-0_outputs.jsonl \
                                  --output_folder $1/repair_sample_1 \
                                  --loc_interval \
                                  --top_n=3 \
                                  --context_window=10 \
                                  --max_samples 5  \
                                  --cot \
                                  --diff_format \
                                  --gen_and_process \
                                  --backend gemini \
                                  --num_threads 2 