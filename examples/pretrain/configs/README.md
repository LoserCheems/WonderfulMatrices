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
| Model | Training Data | Epochs | Steps | Content Length | Tokens | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/JingzeShi/Doge-20M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 2 | 10k | 2048 | 5B | 8e-4 | 0.25M | bfloat16 |
| [Doge-60M](https://huggingface.co/JingzeShi/Doge-60M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 2 | 20k | 2048 | 20B | 6e-4 | 0.5M | bfloat16 |

**Evaluation**:
| Model | TriviaQA | MMLU | ARC | PIQA | HellaSwag | OBQA | Winogrande |
|---|---|---|---|---|---|---|---|
| [Doge-20M](https://huggingface.co/JingzeShi/Doge-20M) | - | 26.01 | 36.15 | 56.26 | 26.60 | 26.60 | 50.12 |
| [Doge-60M](https://huggingface.co/JingzeShi/Doge-60M) | - | 25.81 | 45.49 | 61.37 | 29.65 | 27.40 | 52.57 |

**Environment**:
- Image: nvcr.io/nvidia/pytorch:24.10-py3
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