import os
import logging
import sys
from argparse import ArgumentParser

import yaml
import datasets
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from wonderful_matrices.models import DogeConfig
from wonderful_matrices.models import DogeModel, DogeForCausalLM


logger = logging.getLogger(__name__)

def main(config_path):

    # 获取配置中的参数
    # Get arguments from config
    with open(config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    # 设置日志与输出目录
    # Setup logging and output directory
    model_name = config_path.split('/')[-1].split('.')[0]
    logging_dir = f'{args["logging_dir"]}/{model_name}'
    output_dir = f'{args["output_dir"]}/{model_name}'

    ################################
    # 设置日志
    # Setup Logging
    ################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
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
    dataset = datasets.load_from_disk(args['dataset_path'])
    
    ################################
    # 加载分词器
    # Load tokenizer
    ################################
    tokenizer = AutoTokenizer.from_pretrained(args["model_name_or_path"])

    ################################
    # 加载预训练模型
    # Load pretrained model
    ################################
    logger.info(f"Loading model from {args["model_name_or_path"]}")
    model = AutoModelForCausalLM.from_pretrained(args["model_name_or_path"], trust_remote_code=True)
    model_kwargs = dict(
        torch_dtype=args["torch_dtype"],
        use_cache=True,
    )
    ref_model = model
    ref_model_kwargs = model_kwargs

    model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model structure: {model}")
    logger.info(f"Model parameters: {model_num_params}")
    ref_model_num_params = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
    logger.info(f"Reference model structure: {ref_model}")
    logger.info(f"Reference model parameters: {ref_model_num_params}")

    ################################
    # 设置 DPO 参数
    # Setup DPO arguments
    ################################
    dpo_config = DPOConfig(
        # 日志与输出
        # Logging and Output
        logging_dir=logging_dir,
        report_to=args['report_to'],
        logging_steps=args['logging_steps'],
        output_dir=output_dir,

        # 模型初始化
        # Model initialization
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,

        # 数据处理
        # Dataset processing
        dataset_num_proc=args['preprocessing_num_workers'],
        max_length=args['max_length'],
        max_prompt_length=args['max_prompt_length'],

        # 种子
        # Seed
        seed=args['seed'],

        # 训练设置
        # Training settings
        do_train=args['do_train'],
        num_train_epochs=args['num_train_epochs'],
        per_device_train_batch_size=args['per_device_train_batch_size'],

        # 评估设置
        # Evaluation settings
        do_eval=args['do_eval'],
        eval_strategy=args['eval_strategy'],
        eval_steps=args['eval_steps'],
        per_device_eval_batch_size=args['per_device_eval_batch_size'],
        
        # 学习策略
        # Learning strategy
        optim=args['optim'],
        beta=args['beta'],
        loss_type=args['loss_type'],
        learning_rate=args['learning_rate'],
        lr_scheduler_type=args['lr_scheduler_type'],
        lr_scheduler_kwargs={**args['lr_scheduler_kwargs']},
        warmup_ratio=args['warmup_ratio'],
        weight_decay=args['weight_decay'],
        gradient_accumulation_steps=args['gradient_accumulation_steps'],
        max_grad_norm=args['max_grad_norm'],
        bf16=args['bf16'],

        # 保存策略
        # Save strategy
        save_safetensors=args['save_safetensors'],
        save_strategy=args['save_strategy'],
        save_steps=args['save_steps'],
    )

    ################################
    # 初始化训练器
    # Initialize the trainer
    ################################
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset['dataset_splits'][0],
        eval_dataset=dataset['dataset_splits'][1] if args['do_eval'] else None,
        processing_class=tokenizer,
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

    ################################
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
    
    logger.info("*** Training complete ***")

    ################################
    # 评估
    # Evaluation
    ################################
    if dpo_config.do_eval:
        logger.info("*** Start evaluation... ***")
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(dataset['test'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("*** Evaluation complete ***")
    
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
    tokenizer = AutoTokenizer.from_pretrained(f"{output_dir}")
    model = AutoModelForCausalLM.from_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}-registered")
    model.save_pretrained(f"{output_dir}-registered")
    logger.info(f"Model registered and saved to {output_dir}-registered")

    if args["push_to_hub"] is True:
        logger.info("Pushing to hub...")
        tokenizer.push_to_hub()
        model.push_to_hub()
    
    logger.info("*** Training finished! ***")




if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config_path', type=str, default='./examples/finetune/configs/Doge-20M-Instruct-DPO.yaml', help='path to yaml config file')

    args = arg_parser.parse_args()

    main(args.config_path)
