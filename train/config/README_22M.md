---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smollm-corpus
language:
- en
- zh
pipeline_tag: text-generation
---


# **Doge 22M**

Doge is an ongoing research project where we aim to train a series of small language models to further explore whether the Transformer framework allows for more complex feedforward network structures, enabling the model to have fewer cache states and larger knowledge capacity.

In addition, Doge uses Inner Function Attention with Dynamic Mask as sequence transformation and Cross Domain Mixture of Experts as state transformation. This model is trained by Jingze Shi, it only allows text input and text generation, for detailed algorithm and model architecture, please refer to [Wonderful Matrices](https://arxiv.org/abs/2407.16958), the ongoing research repository is [Doge](https://github.com/LoserCheems/Doge).


## Uses

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("LoserCheems/Doge-22M")
>>> model = AutoModelForCausalLM.from_pretrained("LoserCheems/Doge-22M", trust_remote_code=True)
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
| Doge-22M | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 5k | 2048 | 1B | 8e-4 | 0.25M | bfloat16 |
| Doge-76M | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | 10k | 2048 | 5B | 6e-4 | 0.5M | bfloat16 |


**Training Environment**:
- Image: nvcr.io/nvidia/pytorch:24.10-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers


**Evaluation Results**:

| Model | MMLU | TriviaQA | ARC | PIQA | Hellaswag | OBQA | Wnogrande | Avg |
|-------|------|----------|-----|------|-----------|------|-----------|-----|
| TinyStories-28M | 24.03 | 0.01 | 27.69 | 53.21 | 27.32 | 21.00 | 50.67 | 29.13 |
| **Doge-22M** | 23.11 | 0.00 | 31.77 | 53.10 | 25.29 | 24.40 | 49.56 | 29.60 |
| **Doge-76M** | 23.26 | 0.05 | 37.16 | 56.31 | 27.68 | 27.00 | 49.64 | 31.58 |
| GPT2-137M | 26.29 | 0.49 | 31.09 | 62.51 | 29.76 | 29.40 | 49.72 | 32.75 |
| Pythia-160M | 26.68 | 0.34 | 31.92 | 61.64 | 29.55 | 27.80 | 49.49 | 32.49 |
| SmolLM-135M | 30.23 | 4.11 | 43.99 | 69.60 | 42.30 | 33.60 | 52.70 | 39.50 |

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