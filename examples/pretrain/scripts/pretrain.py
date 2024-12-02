import os
import logging
import sys
from argparse import ArgumentParser

import yaml
import datasets
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from wonderful_matrices.doge.configuration_doge import DogeConfig
from wonderful_matrices.doge.modeling_doge import DogeModel, DogeForCausalLM


logger = logging.getLogger(__name__)

def main(args):

    # 获取配置中的超参数
    # Get hyperparameters from config
    with open(args.config_path, 'r', encoding='utf-8') as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    
    # 设置日志与输出目录
    # Setup logging and output directory
    model_name = args.config_path.split('/')[-1].split('.')[0]
    logging_dir = f'{args.logging_dir}/{model_name}'
    output_dir = f'{args.output_dir}/{model_name}'

    ################################
    # 设置日志
    # Setup Logging
    ################################
    os.makedirs(logging_dir, exist_ok=True)
    file_handler = logging.FileHandler(f'{logging_dir}/log.log')
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, stream_handler],
        level=logging.INFO,
    )
    datasets.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    ################################
    # 加载数据集
    # Load dataset
    ################################
    dataset = datasets.load_from_disk(hyperparameters['training_args']['dataset_path'])
    if hyperparameters['training_args']['per_epoch_max_steps'] != -1:
        # 这样截断的目的是, 指定固定的训练步数, 进行多轮次训练.
        # The purpose of truncating like this is to specify a fixed number of training steps for multiple epochs.
        # 如果进行多节点训练, 需要自行在配置文件中将batch_size * gradient_accumulation_steps / world_size
        # If multi-node training is performed, you need to manually set batch_size * gradient_accumulation_steps / world_size in the configuration file
        dataset["train"] = dataset["train"].select(range(hyperparameters['training_args']['per_epoch_max_steps'] * hyperparameters['training_args']['per_device_train_batch_size'] * hyperparameters['training_args']['gradient_accumulation_steps']))
    logger.info(
        f"Training dataset: {len(dataset['train'])} samples, Evaluation dataset: {len(dataset['test'])} samples."
    )

    ################################
    # 加载分词器
    # Load tokenizer
    ################################
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    ################################
    # 初始化模型
    # Initialize model
    ################################
    logger.info(f"Initializing model from config: {hyperparameters['model_config']}") 
    config = DogeConfig(
        vocab_size=hyperparameters['model_config']['vocab_size'],
        hidden_size=hyperparameters['model_config']['hidden_size'],
        intermediate_size=hyperparameters['model_config']['intermediate_size'],
        num_hidden_layers=hyperparameters['model_config']['num_hidden_layers'],
        hidden_bias=hyperparameters['model_config']['hidden_bias'],
        hidden_dropout=hyperparameters['model_config']['hidden_dropout'],
        hidden_act=hyperparameters['model_config']['hidden_act'],
        max_position_embeddings=hyperparameters['model_config']['max_position_embeddings'],
        rope_theta=hyperparameters['model_config']['rope_theta'],
        use_cache=hyperparameters['model_config']['use_cache'],
        pad_token_id=hyperparameters['model_config']['pad_token_id'],
        bos_token_id=hyperparameters['model_config']['bos_token_id'],
        eos_token_id=hyperparameters['model_config']['eos_token_id'],
        num_attention_heads=hyperparameters['model_config']['num_attention_heads'],
        num_inner_values=hyperparameters['model_config']['num_inner_values'],
        num_inner_value_heads=hyperparameters['model_config']['num_inner_value_heads'],
        num_value_per_head=hyperparameters['model_config']['num_value_per_head'],
        inner_values_retrieval_size=hyperparameters['model_config']['inner_values_retrieval_size'],
        private_expert_retrieval_size=hyperparameters['model_config']['private_expert_retrieval_size'],
        num_cdmmoe_experts=hyperparameters['model_config']['num_cdmmoe_experts'],
        num_cdmmoe_heads=hyperparameters['model_config']['num_cdmmoe_heads'],
        num_cdmmoe_experts_per_head=hyperparameters['model_config']['num_cdmmoe_experts_per_head'],
    )
    config.vocab_size = tokenizer.vocab_size
    model = DogeForCausalLM(config=config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model structure: {model}")
    logger.info(f"Model parameters: {num_params}")

    ################################
    # 设置训练参数
    # Setup training arguments
    ################################
    training_args = TrainingArguments(
        # 随机种子与路径
        # Random seed and paths
        seed=hyperparameters['training_args']['seed'],
        logging_dir=logging_dir,
        logging_steps=hyperparameters['training_args']['logging_steps'],
        output_dir=output_dir,

        # 训练轮次与每个设备的批次
        # Training epochs and per device batch size
        do_train=True,
        num_train_epochs=hyperparameters['training_args']['num_train_epochs'],
        per_device_train_batch_size=hyperparameters['training_args']['per_device_train_batch_size'],

        # 评估策略与评估步数
        # Evaluation strategy and evaluation steps
        do_eval=hyperparameters['training_args']['do_eval'],
        eval_strategy="steps" if hyperparameters['training_args']['do_eval'] else "no",
        eval_steps=hyperparameters['training_args']['eval_steps'],
        per_device_eval_batch_size=hyperparameters['training_args']['per_device_eval_batch_size'],

        # 学习策略
        # Learning strategy
        learning_rate=hyperparameters['training_args']['learning_rate'],
        warmup_ratio=hyperparameters['training_args']['warmup_ratio'],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': hyperparameters['training_args']['min_lr_rate']},
        weight_decay=hyperparameters['training_args']['weight_decay'],

        # 保存策略
        # Save strategy
        save_safetensors=True,
        save_strategy="steps",
        save_steps=hyperparameters['training_args']['save_steps'],

        # 混合精度与梯度累积
        # Mixed precision and gradient accumulation
        bf16=hyperparameters['training_args']['bf16'],
        max_grad_norm=hyperparameters['training_args']['max_grad_norm'],
        gradient_accumulation_steps=hyperparameters['training_args']['gradient_accumulation_steps'],
    )

    ################################
    # 初始化训练器
    # Initialize trainer
    ################################
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.0
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if hyperparameters['training_args']['do_eval'] else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    ################################
    # 训练循环
    # Training loop
    ################################
    logger.info("*** Start training... ***")
    checkpoint = args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics['train_samples'] = len(dataset['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    #################################
    # 保存模型并创建模型卡
    # Save model and create model card
    ################################
    logger.info("*** Saving model... ***")
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(model_name=model_name)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(output_dir)
        logger.info(f"Model card saved to {output_dir}")
    
    logger.info("*** Training finished! ***")

    ################################
    # 评估
    # Evaluation
    ################################
    if training_args.do_eval:
        logger.info("*** Start evaluation... ***")
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(dataset['test'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("*** Evaluation finished! ***")
    
    ################################
    # 注册模型并保存
    # Register the model and save
    ################################
    AutoConfig.register("doge", DogeConfig)
    AutoModel.register(DogeConfig, DogeModel)
    AutoModelForCausalLM.register(DogeConfig, DogeForCausalLM)
    DogeConfig.register_for_auto_class()
    DogeModel.register_for_auto_class("AutoModel")
    DogeForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    tokenizer = AutoTokenizer.from_pretrained(f'{output_dir}')
    model = AutoModelForCausalLM.from_pretrained(f'{output_dir}')
    tokenizer.save_pretrained(f'{output_dir}-registered')
    model.save_pretrained(f'{output_dir}-registered')
    logger.info(f"Model registered and saved to {output_dir}-registered")


if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config_path', type=str, default='./examples/pretrain/configs/Doge-197M.yaml', help='path to yaml config file')
    arg_parser.add_argument('--logging_dir', type=str, default='./logs')
    arg_parser.add_argument('--output_dir', type=str, default='./results')
    arg_parser.add_argument('--tokenizer_path', type=str, default='./examples/tokenizer', help='path to tokenizer')
    arg_parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to checkpoint to resume training")

    args = arg_parser.parse_args()

    main(args)
