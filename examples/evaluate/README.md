# Instructions to evaluate Doge

We use the [lighteval](https://github.com/huggingface/lighteval) toolkit to evaluate the performance of the Doge model.

## Setup

You can install the toolkit using the following command:

```bash
pip install lighteval
```

## Evaluation

如果你是Linux用户, 你可以使用以下命令来评估模型:
en: If you are a Linux user, you can use the following command to evaluate the model:

```bash
bash ./examples/evaluate/eval_downstream_tasks.sh
```

If you are a Windows user, you can use the following command to evaluate the model:

```bash
. ./examples/evaluate/eval_downstream_tasks.ps1
```
> Note: You can modify `MODEL` and `OUTPUT_DIR` in the script to evaluate different models and save the results to different directories.