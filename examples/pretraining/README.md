# Instructions for pre-training Doge

We build the Doge by doing PerTrain on [Smollm-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

## Setup

Follow the instructions in the [README](../README.md) to install the necessary dependencies.

## Datasets

You can download the datasets using the following command:

```bash
python ./examples/pretraining/scripts/download_datasets.py --save_dir ./datasets --cache_dir ./cache --num_proc 1
```

You can preprocess the datasets using the following command:

```bash
python ./examples/pretraining/scripts/preprocess_datasets.py --datasets_dir ./datasets --save_dir ./datasets --tokenizer_path ./examples/tokenizer --train_examples 128000000 --test_examples 1000 --max_length 2048 --num_proc 16
```

>NOTE: Due to the large size of the complete dataset, we only provide a small dataset for demonstration. You can control the size of the dataset by modifying the `--train_examples` and `--test_examples` parameters.

You can concatenate all the sub-datasets using the following command:

```bash
python ./examples/pretraining/scripts/concatenate_datasets.py --datasets_dir ./datasets --save_dir ./datasets --train_examples 128000000 --test_examples 1000 --num_proc 16
```

## Training

We train the Doge-20M on 1 GPU using the following command:

> Note: You can modify the configuration file to implement multi-GPU training, etc.

```bash
python ./examples/pretraining/scripts/pt.py --config_path ./examples/pretraining/configs/Doge-20M.yaml
```

and so on.