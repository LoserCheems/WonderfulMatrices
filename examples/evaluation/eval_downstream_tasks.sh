#!/bin/bash

MODEL="JingzeShi/Doge-20M"
OUTPUT_DIR="./lighteval_results"

lighteval accelerate --override_batch_size 1 \
    --model_args "pretrained=$MODEL" \
    --output_dir $OUTPUT_DIR \
    --tasks "original|mmlu|5|0,lighteval|triviaqa|5|0,lighteval|arc:easy|5|0,leaderboard|arc:challenge|5|0,lighteval|piqa|5|0,leaderboard|hellaswag|5|0,lighteval|openbookqa|5|0,leaderboard|winogrande|5|0"
 