---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smollm-corpus
language:
- en
pipeline_tag: text-generation
---


# **Doge 60M**

Doge is an ongoing research project where we aim to train a series of small language models to further explore whether the Transformer framework allows for more complex feedforward network structures, enabling the model to have fewer cache states and larger knowledge capacity.

In addition, Doge uses Dynamic Mask Attention as sequence transformation and can use Multi-Layer Perceptron or Cross Domain Mixture of Experts as state transformation. Dynamic Mask Attention allows the Transformer to use self-attention during training and state space during inference, and Cross Domain Mixture of Experts can directly inherit the weights of Multi-Layer Perceptron for further training. This model is trained by Jingze Shi, it only allows text input and text generation, for detailed algorithm and model architecture, please refer to [Wonderful Matrices](https://arxiv.org/abs/2412.11834), the ongoing research repository is [Wonderful Matrices](https://github.com/LoserCheems/WonderfulMatrices).


## Uses

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("JingzeShi/Doge-60M")
>>> model = AutoModelForCausalLM.from_pretrained("JingzeShi/Doge-60M", trust_remote_code=True)
>>> inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

>>> out = model.generate(**inputs, max_new_tokens=100)
>>> print(tokenizer.batch_decode(out))
```


## Model Details

> NOTE: This model has not been fine-tuned for instruction

> TODO: The larger model is under training and will be uploaded soon.

**Training**:

| Model | Training Data | Steps | Content Length | Tokens | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/JingzeShi/Doge-20M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 8k  | 2048 | 4B | 8e-3 | 0.5M | bfloat16 |
| [Doge-60M](https://huggingface.co/JingzeShi/Doge-60M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 16k  | 2048 | 16B | 6e-3 | 1M | bfloat16 |

**Evaluation**:

| Model | MMLU | TriviaQA | ARC-E | ARC-C | PIQA | HellaSwag | OBQA | Winogrande | tokens / s on CPU |
|---|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/JingzeShi/Doge-20M) | 25.43 | 0 | 36.83 | 22.53 | 58.38 | 27.25 | 25.60 | 50.20 | 142 |
| [Doge-60M](https://huggingface.co/JingzeShi/Doge-60M) | 26.41 | 0 | 50.00 | 25.34 | 61.43 | 31.45 | 28.00 | 49.64 | 62 |

> All evaluations are done using five-shot settings, without additional training on the benchmarks.

**Procedure**:

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/loser_cheems/huggingface/runs/ydynuvfz) 


**Environment**:

- Image: nvcr.io/nvidia/pytorch:24.12-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers


## Citation

```bibtex
@misc{shi2024wonderfulmatrices,
      title={Wonderful Matrices: Combining for a More Efficient and Effective Foundation Model Architecture}, 
      author={Jingze Shi and Bingheng Wu},
      year={2024},
      eprint={2412.11834},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.11834}, 
}
```