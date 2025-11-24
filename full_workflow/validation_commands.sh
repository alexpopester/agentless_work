#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide an output directory as the first argument."
    exit 1
fi

# Regression
#python -m agentless.test.run_regression_tests --run_id generate_regression_tests \
#                                              --output_file $1/passing_tests.jsonl
# ...

# Reproduction Generation
# samples must match the amount of samples generated
TEST_COUNT=5
TEST_COUNT_MINUS_ONE=$(($TEST_COUNT-1))

#python -m agentless.test.generate_reproduction_tests --max_samples $TEST_COUNT \
#                                                     --output_folder $1/reproduction_test_samples \
#                                                     --num_threads 10 \
#                                                     --backend gemini
#                                                     #--target_id=astropy__astropy-6938 \

# Run Reproduction Test
for i in $(seq 0 $TEST_COUNT_MINUS_ONE)
do
echo "Doing reproduction test: $i"
python -m agentless.test.run_reproduction_tests --run_id="reproduction_test_generation_filter_sample_${i}" \
                                                --test_jsonl="$1/reproduction_test_samples/output_${i}_processed_reproduction_test.jsonl" \
                                                --num_workers 2 \
                                                --testing;
                                                #--instance_id=astropy__astropy-6938 \
done


# Select the best reproduction test
python -m agentless.test.generate_reproduction_tests --max_samples $TEST_COUNT \
                                                     --output_folder $1/reproduction_test_samples \
                                                     --output_file reproduction_tests.jsonl \
                                                     --backend gemini \
                                                     --select
                                                     #--target_id=astropy__astropy-6938 \

# Get the tests after passing
folder=$1/repair_sample_1
for i in $(seq 0 $TEST_COUNT_MINUS_ONE)
do
    run_id_prefix=$(basename $folder); 
    python -m agentless.test.run_reproduction_tests --test_jsonl $1/reproduction_test_samples/reproduction_tests.jsonl \
                                                    --predictions_path="${folder}/output_${i}_processed.jsonl" \
                                                    --run_id="${run_id_prefix}_reproduction_v2_${i}" --num_workers 2;
                                                    #--instance_id=astropy__astropy-6938 \
done
##
###python -m agentless.repair.rerank --patch_folder $1/repair_sample_1/,$1/repair_sample_2/,$1/repair_sample_3/,$1/repair_sample_4/ \
python -m agentless.repair.rerank --patch_folder $1/repair_sample_1/ \
                                  --num_samples 2 \
                                  --deduplicate \
                                  --reproduction
                                  #--target=astropy__astropy-6938 \
##                                 #--regression \