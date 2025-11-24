##!/bin/bash

export GOOGLE_API_KEY="<TODO>"
export HF_TOKEN="<TODO>"
export PROJECT_FILE_LOC="<TODO>"
#LOCALIZE TO SUSPICIOUS FILES

#1
#python -m agentless.fl.localize --file_level \
#                                --output_folder "$1/file_level" \
#                                --backend gemini \
#                                --skip_existing
#                                #--target_id=astropy__astropy-6938 \
#
##2 
#python -m agentless.fl.localize --file_level \
#                                --irrelevant \
#                                --output_folder "$1/file_level_irrelevant" \
#                                --backend gemini \
#                                --skip_existing
#                                #--target_id=astropy__astropy-6938 \
#
##3 (COULD USE GRAPHBERT)
#python -m agentless.fl.retrieve --index_type simple \
#                                --filter_type given_files \
#                                --filter_file "$1/file_level_irrelevant/loc_outputs.jsonl" \
#                                --output_folder $1/retrievel_embedding \
#                                --persist_dir embedding/swe-bench_simple \
#                                --num_threads 1
#                                #--target_id=astropy__astropy-6938
#
##4 (COULD USE GRAPHBERT)
#python -m agentless.fl.combine  --retrieval_loc_file $1/retrievel_embedding/retrieve_locs.jsonl \
#                                --model_loc_file $1/file_level/loc_outputs.jsonl \
#                                --top_n 3 \
#                                --output_folder $1/file_level_combined
#
## LOCALIZE TO RELATED ELEMENTS
#python -m agentless.fl.localize --related_level \
#                                --output_folder $1/related_elements \
#                                --top_n 3 \
#                                --compress_assign \
#                                --compress \
#                                --start_file $1/file_level_combined/combined_locs.jsonl \
#                                --skip_existing \
#                                --backend gemini
#                                #--target_id=astropy__astropy-6938 \
#
## LOCALIZE TO EDIT LOCATIONS
#python -m agentless.fl.localize --fine_grain_line_level \
#                                --output_folder $1/edit_location_samples \
#                                --top_n 3 \
#                                --compress \
#                                --temperature 0.8 \
#                                --num_samples 4 \
#                                --start_file $1/related_elements/loc_outputs.jsonl \
#                                --skip_existing  \
#                                --backend gemini
                                #--target_id=astropy__astropy-6938 \

# LOCALIZE TO INDIVIDUAL LOCATIONS
python -m agentless.fl.localize --merge \
                                --output_folder $1/edit_location_individual \
                                --top_n 3 \
                                --num_samples 4 \
                                --start_file $1/edit_location_samples/loc_outputs.jsonl \
                                --skip_existing  \
                                --backend gemini
                                #--target_id=astropy__astropy-6938 \


