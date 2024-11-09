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

# **Doge 25M**

NOTE: This model is only for testing, more details on the model and training are in the works.

Doge is an ongoing research project where we aim to train a series of small language models to further explore whether the Transformer framework allows for more complex feedforward network structures, enabling the model to have fewer cache states and larger knowledge capacity.

This model is trained by Jingze Shi, it only allows text input and text generation, for detailed algorithm and model architecture, please refer to [Wonderful Matrices](https://arxiv.org/abs/2407.16958v5), the ongoing research repository is [Doge](https://github.com/LoserCheems/Doge).


## Uses

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("LoserCheems/Doge-25M")
>>> model = AutoModelForCausalLM.from_pretrained("LoserCheems/Doge-25M", trust_remote_code=True)
>>> inputs = tokenizer("Hey how are you doing?", return_tensors="pt")

>>> out = model.generate(**inputs, max_new_tokens=10)
>>> print(tokenizer.batch_decode(out))
["Hey how are you doing?\n\nI'm doing great.\n\n"]
```
