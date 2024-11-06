import os
import logging
from argparse import ArgumentParser

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from model.doge.configuration_doge import DogeConfig
from model.doge.modeling_doge import DogeForCausalLM


if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config_path', type=str, default='./model/config/doge_25M.yaml', help='path to yaml config file')
    arg_parser.add_argument('--logging_dir', type=str, default='logs')
    arg_parser.add_argument('--output_dir', type=str, default='results')
    arg_parser.add_argument('--tokenizer_path', type=str, default='./tokenizer', help='path to tokenizer')
    arg_parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to checkpoint to resume training")
    arg_parser.add_argument("--mode", type=str, default='pretrain', choices=['pretrain', 'finetune'], help='pretrain or finetune')

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
    if args.mode == 'pretrain':
        dataset = load_from_disk(hyperparameters['train']['dataset_path'])
    elif args.mode == 'finetune':
        dataset = load_from_disk(hyperparameters['finetune']['dataset_path'])
    dataset["train"] = dataset["train"].select(range(hyperparameters['train']['per_epoch_max_steps'] * hyperparameters['train']['gradient_accumulation_steps']))

    # 加载模型
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.resume_from_checkpoint:
        model = DogeForCausalLM.from_pretrained(args.resume_from_checkpoint)
    else:
        config = DogeConfig(
            vocab_size=hyperparameters['model']['vocab_size'],
            hidden_size=hyperparameters['model']['hidden_size'],
            num_hidden_layers=hyperparameters['model']['num_hidden_layers'],
            hidden_bias=hyperparameters['model']['hidden_bias'],
            hidden_dropout=hyperparameters['model']['hidden_dropout'],
            hidden_act=hyperparameters['model']['hidden_act'],
            max_position_embeddings=hyperparameters['model']['max_position_embeddings'],
            rope_theta=hyperparameters['model']['rope_theta'],
            use_cache=hyperparameters['model']['use_cache'],
            pad_token_id=hyperparameters['model']['pad_token_id'],
            bos_token_id=hyperparameters['model']['bos_token_id'],
            eos_token_id=hyperparameters['model']['eos_token_id'],
            num_attention_heads=hyperparameters['model']['num_attention_heads'],
            num_inner_values=hyperparameters['model']['num_inner_values'],
            cross_domain_intermediate_size=hyperparameters['model']['cross_domain_intermediate_size'],
            private_expert_intermediate_size=hyperparameters['model']['private_expert_intermediate_size'],
            num_cdmmoe_experts=hyperparameters['model']['num_cdmmoe_experts'],
            num_cdmmoe_heads=hyperparameters['model']['num_cdmmoe_heads'],
            num_cdmmoe_experts_per_head=hyperparameters['model']['num_cdmmoe_experts_per_head'],
        )
        config.vocab_size = tokenizer.vocab_size
        model = DogeForCausalLM(config=config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(num_params)

    # 训练参数
    training_args = TrainingArguments(
        seed=233,
        logging_dir=logging_dir,
        logging_steps=hyperparameters['train']['logging_steps'],
        output_dir=output_dir,
        do_train=True,
        num_train_epochs=hyperparameters['train']['num_train_epochs'],
        do_eval=True,
        eval_strategy="steps",
        eval_steps=hyperparameters['train']['eval_steps'],
        per_device_train_batch_size=hyperparameters['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=hyperparameters['train']['per_device_eval_batch_size'],
        weight_decay=hyperparameters['train']['weight_decay'],
        learning_rate=hyperparameters['train']['learning_rate'],
        warmup_ratio=hyperparameters['train']['warmup_ratio'],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': hyperparameters['train']['min_lr_rate']},
        save_safetensors=True,
        save_strategy="steps",
        save_steps=hyperparameters['train']['save_steps'],
        bf16=hyperparameters['train']['bf16'],
        max_grad_norm=hyperparameters['train']['max_grad_norm'],
        gradient_accumulation_steps=hyperparameters['train']['gradient_accumulation_steps'],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)