# TODO: 着手准备SFT的全训练流程


from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

dataset = load_from_disk('./datasets/Infinity-Instruct_gen_processed')
dataset['train'] = dataset['train'].select(range(1000))

if __name__ == "__main__":
    sft_config = SFTConfig(
        seed=233,
        logging_dir="./logs",
        logging_steps=100,
        output_dir="./results",

        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        
        do_eval=True,
        eval_strategy="steps",
        eval_steps=1000,
        per_device_eval_batch_size=1,
        dataset_num_proc=1,

        do_predict=True,
        
        warmup_ratio=0.1,
        learning_rate=6e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': 0.1},

        save_safetensors=True,
        save_strategy="steps",
        save_steps=1000,

        bf16=False,
        max_grad_norm=1.0,
        gradient_accumulation_steps=32,
        max_seq_length=2048,
        num_of_sequences=1,
        packing=False,
    )

    tokenizer = AutoTokenizer.from_pretrained("JingzeShi/Doge-76M")
    model = AutoModelForCausalLM.from_pretrained("JingzeShi/Doge-76M", trust_remote_code=True)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
    )

    trainer.train()