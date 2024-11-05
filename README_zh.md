# Doge

<h4 align="center">
<p>

[English](./README.md) | 简体中文

</p>
</h4>

![Doge](./assets/doge_architecture.png)
> **Wonderful Matrices: More Efficient and Effective Architecture for Language Modeling Tasks**\
> 石竞泽*, 吴冰珩*, 何鹭*, 姜路畅*\
> 论文: [arXiv:2407.16958](https://arxiv.org/abs/2407.16958)

## 关于

本项目是对 [Wonderful Matrices](https://arxiv.org/abs/2407.16958) 论文中, 讨论章节部分的延续研究.

我们希望通过训练 `Doge` 架构的小型语言模型(SLM), 来进一步探索 Transformer 框架是否允许更深更复杂的前馈网络结构, 从而使模型具有更少的缓存状态与更大的知识容量.

并且我们希望能够尽可能使用开源工具和框架, 来简化从处理数据到训练模型的流程, 以便于初学者也能够轻易了解和使用.


## 依赖

- Windows or Linux
- NVIDIA GPU
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+

我们十分建议您安装最新版的 PyTorch 和 CUDA, 以获得最佳的性能.

当然你也可以使用开源的 [Docker PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) 镜像, 来避免配置环境的麻烦.

```bash
docker pull nvcr.io/nvidia/pytorch:24.10-py3
docker run --privileged --gpus all -it --name PyTorch --shm-size=32g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 -v <你的代码路径>:/workspace -v <你的数据集路径>:/workspace/Doge/datasets nvcr.io/nvidia/pytorch:24.10-py3
```

- `pip install transformers`: 后续所有工作的核心框架.
- `pip install datasets sentencepiece boto3`: 用于下载和处理数据集.
- `pip install accelerate`: 用于分布式训练.
- `pip install einx`: CDMoE 模块的快速实现依赖.


## 使用

我们编写了一个 [notebook](./notebook.ipynb)(仍然在更新中) 来展示 数据处理, 模型训练和模型评估的整个流程. 当然你也可以独立使用以下一些模块.

### Inner Function Attention

Doge 模型的序列变换模块.

源代码: [model/modules/innerfuncattn.py](./model/modules/innerfuncattn.py)

使用方法:

```python
import torch
from model.modules.innerfuncattn import InnerFuncAttn

batch, seq_len, dim = 2, 16, 64
x = torch.rand(batch, seq_len, dim)
attention_mask = torch.ones(batch, seq_len)
attn = InnerFuncAttn(
    d_model=dim,
    n_heads=1,
    n_inner_values=1,
    max_position_embeddings=seq_len,
    layer_idx=0,
)
y, past_key_values = attn(x, attention_mask)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

### CDMoE

Doge 模型的状态变换模块.

源代码: [model/modules/cdmoe.py](./model/modules/cdmoe.py)

使用方法:

```python
import torch
from model.modules.cdmoe import CDMoE

batch, seq_len, dim = 2, 16, 64
x = torch.rand(batch, seq_len, dim)
cdmoe = CDMoE(
    d_model=dim,
    act_fn="silu",
    d_cross_domain=dim * 4,
    d_private_expert=dim,
    n_experts=64,
    n_experts_heads=1,
    n_experts_per_head=2,
)
y = cdmoe(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```


## 引用

如果您使用了本代码库, 或者认为我们的工作有价值, 请引用 Doge:

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