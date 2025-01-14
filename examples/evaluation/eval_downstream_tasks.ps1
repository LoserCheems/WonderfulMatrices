
$MODEL = "JingzeShi/Doge-20M"
$OUTPUT_DIR = "./lighteval_results"

if ($MODEL -match "Instruct$") {
    lighteval accelerate --override_batch_size 1 `
    --model_args "pretrained=$MODEL,max_model_length=2048" `
    --output_dir $OUTPUT_DIR `
    --tasks "extended|ifeval|5|0" `
    --use_chat_template `
    --save_details
} else {
    lighteval accelerate --override_batch_size 1 `
    --model_args "pretrained=$MODEL,max_model_length=2048" `
    --output_dir $OUTPUT_DIR `
    --tasks "original|mmlu|5|0,lighteval|triviaqa|5|0,lighteval|arc:easy|5|0,leaderboard|arc:challenge|5|0,lighteval|piqa|5|0,leaderboard|hellaswag|5|0,lighteval|openbookqa|5|0,leaderboard|winogrande|5|0" `
    --save_details
}
