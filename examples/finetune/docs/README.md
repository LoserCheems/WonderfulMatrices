---
library_name: transformers
license: apache-2.0
datasets:
- HuggingFaceTB/smoltalk
base_model:
- JingzeShi/Doge-20M
language:
- en
pipeline_tag: question-answering
---


# **Doge 20M Instruct**

Doge is an ongoing research project where we aim to train a series of small language models to further explore whether the Transformer framework allows for more complex feedforward network structures, enabling the model to have fewer cache states and larger knowledge capacity.

In addition, Doge uses Dynamic Mask Attention as sequence transformation and can use Multi-Layer Perceptron or Cross Domain Mixture of Experts as state transformation. Dynamic Mask Attention allows the Transformer to use self-attention during training and state space during inference, and Cross Domain Mixture of Experts can directly inherit the weights of Multi-Layer Perceptron for further training. This model is trained by Jingze Shi, it only allows text input and text generation, for detailed algorithm and model architecture, please refer to [Wonderful Matrices](https://arxiv.org/abs/2412.11834), the ongoing research repository is [Wonderful Matrices](https://github.com/LoserCheems/WonderfulMatrices).


## Uses

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("JingzeShi/Doge-20M-Instruct")
model = AutoModelForCausalLM.from_pretrained("JingzeShi/Doge-20M-Instruct", trust_remote_code=True)

generation_config = GenerationConfig(
      max_new_tokens=100, 
      use_cache=True, 
      do_sample=True, 
      temperature=0.8, 
      repetition_penalty=1.0
)
steamer = TextStreamer(
      tokenizer=tokenizer, 
      skip_prompt=True
)

prompt = "Hi, how are you doing today?"
conversation = [
      {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=True,
    return_tensors="pt",
)

outputs = model.generate(
    inputs, 
    tokenizer=tokenizer,
    generation_config=generation_config, 
    streamer=steamer
)
```


## Model Details

> TODO: The larger model is under training and will be uploaded soon.

**Training**:
| Model | Training Data | Epochs | Content Length | LR | Batch Size | Precision |
|---|---|---|---|---|---|---|
| [Doge-20M-Instruct](https://huggingface.co/JingzeShi/Doge-20M-Instruct) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 8192 | 8e-5 | 1M | bfloat16 |
| [Doge-60M-Instruct](https://huggingface.co/JingzeShi/Doge-60M-Instruct) | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2 | 8192 | 6e-5 | 1M | bfloat16 |

**Environment**:
- Image: nvcr.io/nvidia/pytorch:24.10-py3
- Hardware: 1x NVIDIA RTX 4090
- Software: Transformers, TRL


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