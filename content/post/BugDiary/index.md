---
title: 踩坑日记
description: "Bug Diary"
url: BugDiary
date: 2023-10-24 00:00:00+0000
categories:
    - Bug
tags:
    - study
    - Bug
    - CS
weight: 1
---

# Bug Diary

## Windows

## Mac

### X11

mac上使用X11与Linux通信，需要：

1. 安装运行[XQuartz](https://www.xquartz.org)，设置 $\rightarrow$ 安全性 $\rightarrow$ 勾选允许从网络客户端链接
2. **运行`xhost +`**，允许所有链接
3. 使用-X连接remote或者在`~/.ssh/config`中`Host`下加入`ForwardX11 yes`
4. remote使用`DISPLAY`环境变量：`export DISPLAY=${HOSTNAME}:0`
5. 真远程机器直接用`ssh`传入命令

## Linux

### vscode通过vpn服务器进行ssh多重跳板

mac通过Easy Connect连接187后将187作为跳板，需要转发v100 vpn服务器的1080端口并且向本地转发，于是本地可以直接通过1080端口的socks代理访问v100服务器，这里不是通过多次ssh跳板（vscode支持性较差），而是通过多次转发vpn端口实现支持。