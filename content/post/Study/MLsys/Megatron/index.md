---
title: Megatron框架学习
description: "Learning Megatron Framework"
slug: megatron-learning
date: 2024-02-07 00:00:00+0000
categories:
    - MLsys
tags:
    - study
    - MLsys
    - CS
weight: 1
---

# Megatron

参考[https://www.cnblogs.com/rossiXYZ/p/15840803.html](https://www.cnblogs.com/rossiXYZ/p/15840803.html)

## 简介

model parallelism:

- 张量并行：通信发生在每层的前向传播和后向传播过程之中，通信类型是all-reduce，不但单次通信数据量大，并且通信频繁。
    - 每个transformer 层内的矩阵乘法被分割到多个GPU上
    - (a) 张量并行所需的all-reduce通信需要通过服务器间的链接，这比多GPU服务器内的高带宽NVLink要慢；
    - (b) 高度的模型并行会产生很多小矩阵乘法（GEMMs），这可能会降低GPU的利用率。
- 流水线并行：通信在流水线阶段相邻的切分点之上，通信类型是P2P通信，单词通信数据量较少但是比较频繁，而且因为流水线的特点，会产生GPU空闲时间，这里称为流水线气泡（Bubble）。

因为张量并行一般都在同一个机器之上，所以通过 NVLink 来进行加速，对于流水线并行，一般通过 Infiniband 交换机进行连接。

最佳micro batch size $b$取决于模型的吞吐量和内存占用特性，以及管道深度$p$、数据并行尺寸$d$和批尺寸$B$

张量模型的并行性在节点（DGX A100服务器）内是最好的，因为它会减少通信量。

尽管数据并行可以带来高效的扩展，但我们不能单独使用数据并行来处理训练批量有限的大型模型，因为a）内存容量不足，b）数据并行的扩展限制（例如，GPT-3的训练批量为1536。因此，数据并行性只支持并行到1536个GPU；然而，大约有10000个GPU用来训练这个模型）

## 整体流程

`pretrain`预训练 + `finetuning`微调

`pretrain`分为`model_provider`获取模型，`train_valid_test_datasets_provider`获取数据集，`forward_step`步进函数，`broadcast_data`广播数据，广播到所有tensor-model-parallel的其他rank上，`get_tensor_model_parallel_src_rank`找到TMP组源节点，作为正确的发送目的

### Pretrain

- 初始化Megatron。
- 使用model_provider设置模型、优化器和lr计划。
- 调用train_val_test_data_provider以获取train/val/test数据集。
- 使用forward_step_func训练模型。

#### 初始化Megatron

##### 初始化全局变量

假定目前有16个GPU，属于两个node，rank 0 ～7 属于第一个节点，rank 8 ～ 15 属于第二个节点。下面的 gi 指的是第 i 个 GPU。

- `_TENSOR_MODEL_PARALLEL_GROUP` ：当前 rank 所属于的Intra-layer model parallel group，就是tensor 并行进程组。
    - 假如每一层分为两个tensor，则 `_TENSOR_MODEL_PARALLEL_GROUP` 例子为：[g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]。
- `_PIPELINE_MODEL_PARALLEL_GROUP` ：当前 rank 所属于的Intra-layer model parallel group，就是流水线进程组。
    - 假如流水线深度为4，则例子为 [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]。
- `_MODEL_PARALLEL_GROUP` ：当前 rank 所属于的模型并行进程组，包括了以上两组。
    - 针对我们例子，就是完整模型被复制了两份，两份分别对应的 GPU 具体是[0, 1, 4, 5, 8, 9, 12, 13]，[2, 3, 6, 7, 10, 11, 14, 15]
- `_EMBEDDING_GROUP` ： 嵌入对应的进程组。
- `_DATA_PARALLEL_GROUP` ：当前 rank 所属于的Data parallel group。
    - 假如数据并行度数为2，则例子为[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]。
    - 有点疑惑

### Parallel Transformer Layer

#### 初始化

- 生成一个LayerNorm处理输入数据。
- 生成并行Attention。
- 生成处理attention输出的LayerNorm。
- 如果是decoder，则生成一个ParallelAttention。
- 生成一个并行MLP。

- ColumnParallelLinear
- RowParallelLinear

##### 命名规范

- h: hidden size
- n: number of attention heads
- p: number of model parallel partitions
- np: n/p
- hp: h/p
- hn: h/n
- b: batch size
- s: sequence length
- l: number of layers
- Transformer 的输入size是 [s, b, h]，返回一个同样size的张量，我们使用 hyperparameters 作为transformer 的超参数。

```python
class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear( # 列切分
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False, # 这里是false，采用第二种方案
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion # gelu
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear( # 行切分
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

```

## 整体架构

