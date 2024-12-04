#!/bin/bash

MODEL="./results/Doge_Eval/197M/checkpoint-10000"
OUTPUT_DIR="./lighteval_results"

lighteval accelerate --override_batch_size 16 \
    --model_args "pretrained=$MODEL" \
    --output_dir $OUTPUT_DIR \
    --tasks "leaderboard|mmlu|0|1,lighteval|triviaqa|0|1,lighteval|arc:easy|0|1,lighteval|piqa|0|1,leaderboard|hellaswag|0|1,lighteval|openbookqa|0|1,leaderboard|winogrande|0|1"
 