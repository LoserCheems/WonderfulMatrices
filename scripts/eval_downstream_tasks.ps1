
$MODEL = "./results/Doge_Eval/197M/checkpoint-10000"
$OUTPUT_DIR = "./lighteval_results"


lighteval accelerate --override_batch_size 16 `
    --model_args "pretrained=$MODEL" `
    --output_dir $OUTPUT_DIR `
    --tasks "lighteval|arc:easy|0|1,leaderboard|arc:challenge|0|1,leaderboard|hellaswag|0|1,lighteval|piqa|0|1,lighteval|openbookqa|0|1,lighteval|triviaqa|0|1,lighteval|triviaqa|5|1,leaderboard|winogrande|0|1,leaderboard|mmlu:abstract_algebra|0|1,leaderboard|mmlu:anatomy|0|1,leaderboard|mmlu:astronomy|0|1,leaderboard|mmlu:business_ethics|0|1,leaderboard|mmlu:clinical_knowledge|0|1,leaderboard|mmlu:college_biology|0|1,leaderboard|mmlu:college_chemistry|0|1,leaderboard|mmlu:college_computer_science|0|1,leaderboard|mmlu:college_mathematics|0|1,leaderboard|mmlu:college_medicine|0|1,leaderboard|mmlu:college_physics|0|1,leaderboard|mmlu:computer_security|0|1,leaderboard|mmlu:conceptual_physics|0|1,leaderboard|mmlu:econometrics|0|1,leaderboard|mmlu:electrical_engineering|0|1,leaderboard|mmlu:elementary_mathematics|0|1,leaderboard|mmlu:formal_logic|0|1,leaderboard|mmlu:global_facts|0|1,leaderboard|mmlu:high_school_biology|0|1,leaderboard|mmlu:high_school_chemistry|0|1,leaderboard|mmlu:high_school_computer_science|0|1,leaderboard|mmlu:high_school_european_history|0|1,leaderboard|mmlu:high_school_geography|0|1,leaderboard|mmlu:high_school_government_and_politics|0|1,leaderboard|mmlu:high_school_macroeconomics|0|1,leaderboard|mmlu:high_school_mathematics|0|1,leaderboard|mmlu:high_school_microeconomics|0|1,leaderboard|mmlu:high_school_physics|0|1,leaderboard|mmlu:high_school_psychology|0|1,leaderboard|mmlu:high_school_statistics|0|1,leaderboard|mmlu:high_school_us_history|0|1,leaderboard|mmlu:high_school_world_history|0|1,leaderboard|mmlu:human_aging|0|1,leaderboard|mmlu:human_sexuality|0|1,leaderboard|mmlu:international_law|0|1,leaderboard|mmlu:jurisprudence|0|1,leaderboard|mmlu:logical_fallacies|0|1,leaderboard|mmlu:machine_learning|0|1,leaderboard|mmlu:management|0|1,leaderboard|mmlu:marketing|0|1,leaderboard|mmlu:medical_genetics|0|1,leaderboard|mmlu:miscellaneous|0|1,leaderboard|mmlu:moral_disputes|0|1,leaderboard|mmlu:moral_scenarios|0|1,leaderboard|mmlu:nutrition|0|1,leaderboard|mmlu:philosophy|0|1,leaderboard|mmlu:prehistory|0|1,leaderboard|mmlu:professional_accounting|0|1,leaderboard|mmlu:professional_law|0|1,leaderboard|mmlu:professional_medicine|0|1,leaderboard|mmlu:professional_psychology|0|1,leaderboard|mmlu:public_relations|0|1,leaderboard|mmlu:security_studies|0|1,leaderboard|mmlu:sociology|0|1,leaderboard|mmlu:us_foreign_policy|0|1,leaderboard|mmlu:virology|0|1,leaderboard|mmlu:world_religions|0|1"