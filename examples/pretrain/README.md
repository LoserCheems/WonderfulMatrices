# Instructions to train Doge

We build the Doge by doing PerTrain on [Smollm-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

## Setup

Follow the instructions in the [README](../README.md) to install the necessary dependencies.

## Datasets

You can download the datasets using the following command:

```bash
python ./examples/pretrain/scripts/download_datasets.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

You can preprocess the datasets using the following command:

```bash
python ./examples/pretrain/scripts/preprocess_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_path ./examples/tokenizer --tokens 100000000000 --max_length 2048 --num_proc 16
```

You can concatenate all the sub-datasets using the following command:

```bash
python ./examples/pretrain/scripts/concatenate_datasets.py --datasets_dir ./datasets --save_dir ./datasets --num_proc 16
```

## Training

We train the Doge-22M on 1 GPU using the following command:

> Note: You can modify the configuration file to implement multi-GPU training, etc.

```bash
python ./examples/pretrain/scripts/pretrain.py --config_path ./examples/pretrain/configs/doge_22M.yaml --logging_dir ./logs --output_dir ./results --tokenizer_path ./examples/tokenizer --resume_from_checkpoint <path_to_checkpoint>
```

and so on.