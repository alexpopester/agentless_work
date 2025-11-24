#!/bin/bash

results_dir="full_results_with_semantic_scoring/swe-bench-lite"

/bin/bash full_workflow/localization_commands_with_semantic_scoring.sh $results_dir

/bin/bash full_workflow/repair_commands.sh $results_dir

/bin/bash full_workflow/validation_commands.sh $results_dir