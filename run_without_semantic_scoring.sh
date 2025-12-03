#!/bin/bash
# IF using APIS
export GOOGLE_API_KEY="<TODO>"

# IF USING LOCAL MODEL
#export PATH_TO_LOCAL_MODEL="/home/alex/agentless/llama-3.1-8b.gguf"

#export PROJECT_FILE_LOC="/home/alex/agentless/repo_structure/repo_structures"
export PROJECT_FILE_LOC="/home/apope/documents/neural-networks/swebenchlite/SWE-bench_Lite/repo_structure/repo_structures"

results_dir="full_results_without_semantic_scoring/swe-bench-lite"

/bin/bash full_workflow/localization_commands.sh $results_dir

#/bin/bash full_workflow/repair_commands.sh $results_dir

#/bin/bash full_workflow/validation_commands.sh $results_dir