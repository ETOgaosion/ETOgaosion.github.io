---
title: 论文笔记 - Bamboo：Making Preemptible Instances Resilient for Affordable Training of Large DNNs
description: Article Note - Bamboo - Making Preemptible Instances Resilient for Affordable Training of Large DNNs
slug: research-mlsys-Bamboo
date: 2024-03-03 14:55:00+0000
categories:
    - MLsys
tags:
    - research
    - MLsys
    - CS
    - ArticleNotes
weight: 1
---
# Bamboo：Making Preemptible Instances Resilient for Affordable Training of Large DNNs


经过git version二分搜索，确定Bamboo是基于Deepspeed v0.5.2左右版本修改的，主要修改内容和论文算法位于`/deepspeed/runtime/pipe/engine.py`

## Questions

1. `num_stages`/`nnodes`之间的关系，流水线并行使得机器数量在超过stage数量时才有冗余，但运行代码时发现系统运行只需要启动`num_stage`数量的机器，剩余的机器会阻塞在rdzv配置环节。如何实现动态加入机器
2. 目前尚无法`kill`超过1个节点，会出现通信中断问题，NCCL直接报错退出
3. `Simulator`有的模型没有`steps_per_run`参数，导致必须设置`duration`强制退出

## Bamboo Simulator设计源码阅读

### Trace数据

首先来看一下trace数据（可能是Google Cloud）：

![gloud-trace](photos/gcloud-tace.png)

但实际上Simulator只用了两个Amazon EC2两种节点

Amazon EC2节点的数据为csv格式，三列为delta(ms), action(kind), node，意思是node采取某种行动的时间点

### 流程解释

本质上是状态机转换，通过维护events队列对整个运作流程进行模拟，有以下几种状态，小写的状态为函数：

- `SPOT_INSTANCE_ADD`: 增加spot instance节点
    - 时间增加`spot_instance_creation_time`
- `SPOT_INSTANCE_READY`：增加节点后，节点进入ready状态
    - 集群未运行：`simulate_rendezvous_start`
    - 集群在rdzv配置：加入配置
    - 集群已经运行：加入等待
- `GLOBAL_RENDEZVOUS_TIMEOUT`：进入ready状态后，进入global rdzv配置阶段
- `SPOT_INSTANCE_REMOVE`: 减少spot instance节点，需要fallback_event，源代码直接将单步时长$\times$一个固定的`fallback_slowdown`
    - `simulate_fatal_failure`：这个node已经恢复过别人，没人有他所包含的所有坐标
- `TRAINING_STEP_COMPLETE`，在`GLOBAL_RENDEZVOUS_TIMEOUT`之后完成对单步执行delta的模拟后结束单步，需要处理fallback_event即减少节点，以及增加节点，前往：
- `simulate_should_reconfigure`：判断pipeline是否更宽
    - 若更宽，进入`LOCAL_RENDEZVOUS_TIMEOUT`，但与global似乎没有区别



### 模拟策略

现有数据：

- `num_stages`:$2$，`pipeline`:$1\rightarrow 2\rightarrow 3\rightarrow 4$

首先需要对采到的数据进行拟合：

- 节点数量、pipeline宽度与num_stages对应的delta
- 增加节点状态改变本身带来的开销
- 减少节点减速比
- `delta_reconfigure_class_time`
- `step_delta`