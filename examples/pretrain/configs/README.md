---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smollm-corpus
language:
- en
pipeline_tag: text-generation
---


# **Doge 76M**

Doge is an ongoing research project where we aim to train a series of small language models to further explore whether the Transformer framework allows for more complex feedforward network structures, enabling the model to have fewer cache states and larger knowledge capacity.

In addition, Doge uses Inner Function Attention with Dynamic Mask as sequence transformation and Cross Domain Mixture of Experts as state transformation. This model is trained by Jingze Shi, it only allows text input and text generation, for detailed algorithm and model architecture, please refer to [Wonderful Matrices](https://arxiv.org/abs/2407.16958), the ongoing research repository is [Doge](https://github.com/LoserCheems/Doge).


## Uses

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("JingzeShi/Doge-76M")
>>> model = AutoModelForCausalLM.from_pretrained("JingzeShi/Doge-76M", trust_remote_code=True)
>>> inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

>>> out = model.generate(**inputs, max_new_tokens=100)
>>> print(tokenizer.batch_decode(out))
```


## Model Details
> NOTE: This model has not been fine-tuned for instruction
> TODO: The larger model is under training and will be uploaded soon.

**Model Architecture**: The model architecture is a Transformer with Inner Function Attention with Dynamic Mask as sequence transformation and Cross Domain Mixture of Experts as state transformation. It can be simply understood as a Transformer with all attention and feedforward layers being sparse activation structures. For detailed information on the architecture, please refer to [Wonderful Matrices](https://arxiv.org/abs/2407.16958).

|| Training Data | Steps | Content Length | Tokens | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|---|
| [Doge-22M](https://huggingface.co/LoserCheems/Doge-22M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 5k | 2048 | 1B | 8e-4 | 0.25M | bfloat16 |
| [Doge-76M](https://huggingface.co/JingzeShi/Doge-76M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 10k | 2048 | 5B | 6e-4 | 0.5M | bfloat16 |
| [Doge-197M](https://huggingface.co/JingzeShi/Doge-197M) | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 20k | 2048 | 20B | 5e-4 | 1M | bfloat16 |


**Training Environment**:
- Image: nvcr.io/nvidia/pytorch:24.10-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers

## Citation

```bibtex
@misc{shi2024wonderfulmatrices,
      title={Wonderful Matrices: More Efficient and Effective Architecture for Language Modeling Tasks}, 
      author={Jingze Shi and Bingheng Wu and Lu He and Luchang Jiang},
      year={2024},
      eprint={2407.16958},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.16958}, 
}
```