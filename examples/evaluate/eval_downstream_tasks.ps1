
$MODEL = "./results/Doge_Eval/197M/checkpoint-13400"
$OUTPUT_DIR = "./lighteval_results"

lighteval accelerate --override_batch_size 16 `
    --model_args "pretrained=$MODEL" `
    --output_dir $OUTPUT_DIR `
    --tasks "lighteval|arc:easy|0|1,leaderboard|arc:challenge|0|1,leaderboard|hellaswag|0|1,original|mmlu|0|1,lighteval|piqa|0|1,lighteval|openbookqa|0|1,lighteval|triviaqa|0|1,lighteval|triviaqa|5|1,leaderboard|winogrande|0|1"