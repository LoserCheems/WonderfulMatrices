import os
import logging
from argparse import ArgumentParser

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--pretrained_model_name_or_path', type=str, default='JingzeShi/Doge-76M', help='pretrained model name or path')
    arg_parser.add_argument('--config_path', type=str, default='./examples/pretrain/configs/doge_197M.yaml', help='path to yaml config file')
    arg_parser.add_argument('--logging_dir', type=str, default='logs')
    arg_parser.add_argument('--output_dir', type=str, default='results')
    arg_parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume from checkpoint")

    args = arg_parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    model_name = args.config_path.split('/')[-1].split('.')[0]
    logging_dir = f'{args.logging_dir}/{model_name}'
    output_dir = f'{args.output_dir}/{model_name}'

    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(filename=f'{logging_dir}/log.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # 加载数据集
    # Load dataset
    dataset = load_from_disk(hyperparameters['training_args']['dataset_path'])
    dataset["train"] = dataset["train"].select(range(hyperparameters['training_args']['per_epoch_max_steps'] * hyperparameters['training_args']['per_device_train_batch_size'] * hyperparameters['training_args']['gradient_accumulation_steps']))

    # 加载模型
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, trust_remote_code=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(f"Model parameters: {num_params}")

    # 监督微调参数
    # supervised finetuning argements
    sft_config = SFTConfig(
        seed=233,
        logging_dir=logging_dir,
        logging_steps=hyperparameters['training_args']['logging_steps'],
        output_dir=output_dir,

        do_train=True,
        num_train_epochs=hyperparameters['training_args']['num_train_epochs'],
        per_device_train_batch_size=hyperparameters['training_args']['per_device_train_batch_size'],
        
        do_eval=True,
        eval_strategy="steps",
        eval_steps=hyperparameters['training_args']['eval_steps'],
        per_device_eval_batch_size=hyperparameters['training_args']['per_device_eval_batch_size'],
        dataset_num_proc=hyperparameters['training_args']['dataset_num_proc'],
        
        learning_rate=hyperparameters['training_args']['learning_rate'],
        warmup_ratio=hyperparameters['training_args']['warmup_ratio'],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': hyperparameters['training_args']['min_lr_rate']},
        weight_decay=hyperparameters['training_args']['weight_decay'],

        save_safetensors=True,
        save_strategy="steps",
        save_steps=hyperparameters['training_args']['save_steps'],

        bf16=hyperparameters['training_args']['bf16'],
        max_grad_norm=hyperparameters['training_args']['max_grad_norm'],
        gradient_accumulation_steps=hyperparameters['training_args']['gradient_accumulation_steps'],
        max_seq_length=hyperparameters['training_args']['max_seq_length'],
        packing=hyperparameters['training_args']['packing'],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)