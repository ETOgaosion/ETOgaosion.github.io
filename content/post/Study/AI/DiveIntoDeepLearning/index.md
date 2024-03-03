---
title: Dive Into Deep Learning - 笔记
description: "Notes about Dive Into Deep Learning"
slug: dive-into-deep-learning-notes
date: 2024-02-09 00:00:00+0000
categories:
    - AI
tags:
    - study
    - AI
    - CS
    - Notes
weight: 1
---

# Dive Into Deep Learning

## Math

### 梯度

$n$维向量$\textbf{x}=[x_{1}, \cdots, x_{n}]^{T}$，函数$f(x)$为$\mathbb{R}^{n}\rightarrow \mathbb{R}$，$f(\textbf{x})$相对于$\textbf{x}$的梯度：

$\nabla_{\textbf{x}}f(\textbf{x})=[\frac{\partial f(\textbf{x})}{\partial x_{1}}, \cdots, \frac{\partial f(\textbf{x})}{\partial x_{n}}]^{T}$

规则：

- 对所有$\textbf{A} \in \mathbb{R}^{m\times n}$，都有$\nabla_{\textbf{x}}\textbf{Ax}=\textbf{A}^{T}$
- 对所有$\textbf{A} \in \mathbb{R}^{n\times m}$，都有$\nabla_{\textbf{x}}\textbf{x}^{T}\textbf{A}=\textbf{A}$
- 对所有$\textbf{A} \in \mathbb{R}^{n\times n}$，都有$\nabla_{\textbf{x}}\textbf{x}^{T}\textbf{Ax}=(\textbf{A}+\textbf{A}^{T})\textbf{x}$
- $\nabla_{\textbf{x}}||\textbf{x}||^{2} = 2\textbf{x}$

对于前向传播和反向传播的理解见[机器之心文章](https://www.jiqizhixin.com/graph/technologies/7332347c-8073-4783-bfc1-1698a6257db3)