# Instructions for fine-tuning Doge to Doge-Instruct

We build the Doge-Instruct by first SFT on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) and then DPO on [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

## Setup

Follow the instructions in the [README](../README.md) to install the necessary dependencies.

## Datasets

You can download the datasets using the following command:

```bash
python ./examples/finetuning/scripts/download_datasets.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

Then preprocess the datasets using the following command:

```bash
python ./examples/finetuning/scripts/preprocess_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_path JingzeShi/Doge-20M --num_proc 8
```

## Training

We use the following command to SFT Doge-20M on 1 GPU:

```bash
python ./examples/finetuning/scripts/sft.py --pretrained_model_name_or_path JingzeShi/Doge-20M --config_path ./examples/finetuning/configs/Doge-20M-Instruct.yaml --logging_dir ./logs --output_dir ./results --resume_from_checkpoint <path_to_checkpoint>
```

Then we use the following command to DPO Doge-20M-SFT on 1 GPU:

```bash
python ./examples/finetuning/scripts/dpo.py --config_path ./examples/finetuning/configs/Doge-20M-Instruct-DPO.yaml
```

> Note: You can modify the configuration file to implement multi-GPU training, etc.

and so on.