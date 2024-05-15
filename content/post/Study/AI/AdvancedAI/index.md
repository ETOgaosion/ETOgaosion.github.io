---
title: 高级人工智能 - 课程笔记
description: "Course Notes about Advanced AI"
slug: advanced-ai-course-notes
date: 2024-05-13 00:00:00+0000
categories:
    - AI
tags:
    - study
    - AI
    - CS
    - Notes
weight: 1
---

# 高级人工智能 - 课程笔记

## 搜索

- 完备性：是否能够找到存在的结果
- 时间复杂度
- 空间复杂度
- 最优性：发现的结果是否是代价最小的

复杂度由三个量表达：

- 分子因子`b`：搜索树中结点的最大分支数
- 深度`d`：目标节点所在的最浅深度
- `m`：任何路径的最大长度

- 盲目搜索
- 启发式搜索：知道非目标状态是否比其他状态更有希望接近目标

![blind search](photos/blind_search.png)

### A\*搜索

评估函数：

$$f(n)=g(n)+h(n)$$

- $g(n)$ 从开始节点到n的代价
- $h(n)$从节点n到目标结点的最小代价路径的估计值

可采纳启发式(Admissible Heuristic)：$h(n)\leq h^{*}(n)$，$h^{*}(n)$为$n$到目标结点的真正代价，即永远不会高估真实的代价

一致的启发式(Consistent Heuristic)：$h(n)\leq c(n, a, n’) + h(n’)$，到当前节点到目标的代价不大于后继节点的单步代价与后继节点到目标的估计代价之和，即$f(n)$非递减

一致的图搜索$\rightarrow$最优的