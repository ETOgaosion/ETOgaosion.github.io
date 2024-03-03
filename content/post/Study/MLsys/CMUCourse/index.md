---
title: MLsys - CMU课程笔记
description: "CMU Course Notes about MLsys"
slug: CMU-MLsys-course-notes
date: 2024-02-07 00:00:00+0000
categories:
    - MLsys
tags:
    - study
    - MLsys
    - CS
    - Course Notes
weight: 1
---

# Machine Learning System

## Intro

ML systems stacks:

| Stacks                               | Explanation                                |
| ------------------------------------ | ------------------------------------------ |
| Automatic Differentiation            | 自动生成反向传播计算图并计算梯度           |
| Graph-Level Optimization             | 优化计算图，应用数学变换                   |
| Parallelization/Distributed Training | 决定如何在分布式异构集群中最大化并行       |
| Data Layout and Placement            | 如何在内存层级中放置张量，使用何种数据分布 |
| Kernel Optimizations                 | 生成高性能Kernel和不同硬件后端的可执行文件 |
| Memory Optimizations                 | 最小化AI硬件ML运算的内存需求               |



