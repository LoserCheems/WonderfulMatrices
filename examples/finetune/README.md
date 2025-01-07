# Instructions to train Doge-Instruct

我们通过首先在[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)上进行SFT, 然后在 [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 上进行DPO, 来构建 Doge-Instruct.
We build the Doge-Instruct by first SFT on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) and then DPO on [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

## Setup

Follow the instructions in the [README](../README.md) to install the necessary dependencies.

## Datasets

You can download the datasets using the following command:

```bash
python ./examples/finetune/scripts/download_datasets.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

Then preprocess the datasets using the following command:

```bash
python ./examples/finetune/scripts/preprocess_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_path JingzeShi/Doge-20M --num_proc 8
```

## Training

We train the Doge-20M-Instruct on 1 GPU using the following command:

> Note: You can modify the configuration file to implement multi-GPU training, etc.

```bash
python ./examples/finetune/scripts/sft.py --pretrained_model_name_or_path JingzeShi/Doge-20M --config_path ./examples/finetune/configs/Doge-20M-Instruct.yaml --logging_dir ./logs --output_dir ./results --resume_from_checkpoint <path_to_checkpoint>
```

and so on.