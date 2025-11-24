#!/bin/bash

results_dir="results_without_semantic_scoring/swe-bench-lite"

/bin/bash full_workflow/localization_commands.sh $results_dir

/bin/bash full_workflow/repair_commands.sh $results_dir

/bin/bash full_workflow/validation_commands.sh $results_dir