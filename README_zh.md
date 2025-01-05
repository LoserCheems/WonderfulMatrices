<!-- coding=utf-8
Copyright 2024 Jingze Shi and Bingheng Wu. All rights reserved.

This code is based on the Wonderful Matrices paper implementation.

    https://arxiv.org/abs/2412.11834

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->


# Wonderful Matrices

<h4 align="center">
<p>

[English](./README.md) | 简体中文

</p>
</h4>

![Wonderful_Matrices](./assets/wonderful_matrices.png)
> **Wonderful Matrices: More Efficient and Effective Architecture for Language Modeling Tasks**\
> 石竞泽*, 吴冰珩*\
> 论文: [arXiv:2412.11834](https://arxiv.org/abs/2412.11834)


## 关于

本项目除了提供了 [Wonderful Matrices](https://arxiv.org/abs/2412.11834) 论文中模块和架构的实现代码外, 还是讨论章节部分的延续研究.

`Doge` 架构是使用 `动态掩码自注意力` 的Transformer模型, 它可以理解为在训练时使用自注意力来训练, 并使用状态空间来推理.

我们希望通过训练 `Doge` 架构的小型语言模型(SLM), 来进一步探索 Transformer 框架是否允许更深更复杂的前馈网络结构, 从而使模型具有更少的缓存状态与更大的知识容量.

并且我们希望能够尽可能使用开源工具和框架, 来简化从处理数据到训练模型的流程, 以便于初学者也能够轻易了解和使用.


## 依赖

- Windows or Linux
- NVIDIA GPU
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

我们十分建议您安装最新版的 PyTorch 和 CUDA, 以获得最佳的性能.

当然你也可以使用开源的 [Docker PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) 镜像, 来避免配置环境的麻烦.

```bash
docker pull nvcr.io/nvidia/pytorch:24.12-py3
docker run --privileged --gpus all -it --name PyTorch --shm-size=32g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 -v <你的代码路径>:/workspace -v <你的数据集路径>:/workspace/Doge/datasets nvcr.io/nvidia/pytorch:24.12-py3
```

- `pip install transformers`: 后续所有工作的核心框架.
- `pip install datasets sentencepiece boto3`: 用于下载和处理数据集.
- `pip install accelerate`: 用于分布式训练.
- `pip install einx`: CDMoE 模块的快速实现依赖.

## 安装

```bash
git clone https://github.com/LoserCheems/WonderfulMatrices.git
cd WonderfulMatrices
pip install -e .
```

## 使用

我们编写了一个 [notebook](./examples/notebook.ipynb)(仍然在更新中) 来展示 数据处理, 模型训练和模型评估的整个流程. 你可以使用以下完整架构或者单个模块.

### Cheems 架构

![Cheems](./assets/cheems_architecture.png)

Cheems 架构的建模代码.

源代码: [modeling_cheems.py](./src/wonderful_matrices/models/modeling_cheems.py)

使用方法:

```python
import torch
from wonderful_matrices.model.configuration_cheems import CheemsConfig
from wonderful_matrices.model.modeling_cheems import CheemsForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<your_model_path_or_name>")
config = CheemsConfig()
model = CheemsForCausalLM(config)
input_ids = tokenizer("Hi, how are you today?", return_tensors="pt")
outputs = model.generate(**input_ids, max_length=100)
print(tokenizer.batch_decode(outputs))
```

### Doge 架构

![Doge](./assets/doge_architecture.png)

Doge 架构的建模代码.

源代码: [modeling_doge.py](./src/wonderful_matrices/models/modeling_doge.py)

使用方法:

```python
import torch
from wonderful_matrices.model.configuration_doge import DogeConfig
from wonderful_matrices.model.modeling_doge import DogeForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<your_model_path_or_name>")
config = DogeConfig()
model = DogeForCausalLM(config)
input_ids = tokenizer("Hi, how are you today?", return_tensors="pt")
outputs = model.generate(**input_ids, max_length=100)
print(tokenizer.batch_decode(outputs))
```

### 动态掩码注意力 模块

![DMAttn](./assets/dmattn.png)
![DMAttn](./assets/mqar.png)

Doge 模型的序列变换模块.

源代码: [dmattn.py](./src/wonderful_matrices/modules/dmattn.py)

使用方法:

```python
import torch
from wonderful_matrices.modules.dmcattn import DMAttn

batch, seq_len, dim = 2, 16, 64
x = torch.rand(batch, seq_len, dim)
attention_mask = torch.ones(batch, seq_len)
attn = DMAttn(
    d_model=dim,
    n_heads=1,
    max_position_embeddings=seq_len,
    layer_idx=0,
)
y, past_key_values = attn(x, attention_mask)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

### 交叉领域混合专家 模块

![CDMoE](./assets/cdmoe.png)
![CDMoE](./assets/merm.png)

Doge 模型的状态变换模块.

源代码: [cdmoe.py](./src/wonderful_matrices/modules/cdmoe.py)

使用方法:

```python
import torch
from wonderful_matrices.modules.cdmoe import CDMoE

batch, seq_len, dim = 2, 16, 64
x = torch.rand(batch, seq_len, dim)
cdmoe = CDMoE(
    d_model=dim,
    act_fn="silu",
    d_ff=dim * 4,
    d_private_expert_retrieval=64,
    n_experts=64,
    n_experts_heads=1,
    n_experts_per_head=2,
)
y = cdmoe(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```


## 引用

如果您使用了本代码库, 或者认为我们的工作有价值, 请引用我们的论文:

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
