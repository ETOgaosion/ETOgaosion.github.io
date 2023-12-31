---
title: 计算机网络 - 笔记
description: "Notes about Computer Network"
slug: network-other-book-note
date: 2022-06-28 00:00:00+0000
categories:
    - Computer Network
tags:
    - study
    - Computer Network
    - CS
weight: 1
---

# 网络 note

## 浏览器

### 首先解析url

http方法：

|方法|含义|
|:-:|:-:|
|GET|获取指定信息，uri为文件则返回内容，uri为CGI程序，返回输出|
|POST|客户端向服务端发送表单数据|
|HEAD|与GET基本相同，只返回http小洗头，获取文件最后更新时间等属性信息|
|OPTIONS|通知或查询通信选项|
|PUT|替换uri指定的服务器的文件，若指定的文件不存在则创建|
|DELETE|删除uri指定的服务器文件|
|TRACE|将服务器收到的请求行和头部直接返回客户端|
|CONNECT|使用代理传输加密消息时使用的方法|

http状态码：

|状态码|含义|
|:-:|:-:|
|1xx|告知请求处理进度和情况|
|2xx|成功|
|3xx|表示需要进一步操作|
|4xx|客户端错误|
|5xx|服务器错误|

### 向DNS服务器查询Web服务器的IP地址

IP地址，32bit数字，按照8bit一组分为4组

附加子网掩码，一串与IP地址相同的32bit数字，左边一半都是1，右边一半都是0，子网掩码为1的部分表示网络号，0的部分表示主机号

可以采用网络号bit数直接表示掩码，主机号全Bit为0则为子网，主机号部分全Bit为1则表示对整个子网进行广播

向DNS服务器查询IP地址：域名解析，解析器（包含在Socket库中）

DNS服务器接力查找，通过缓存加快

### 委托协议栈发送消息

- 创建套接字
- 将管道连接到服务器端的套接字上
- 收发数据
- 断开管道并删除套接字

![TCP_IP_layers](photos/TCP_IP_layers.png)

> 浏览器、邮件等一般应用程序收发数据时用 TCP; 
> DNS 查询等收发较短的控制数据时用 UDP。

协议栈上半部分TCP, UDP
下半部分用IP协议控制网络包收发操作的部分，IP包括ICMP和ARP协议，前者告知网络包传送过程产生的错误及各种控制信息，ARP根据IP地址查询相应以太网MAC地址

套接字：OS，给应用程序描述符
连接：通信双方交换控制信息，填充到套接字相应字段

网络包中控制信息：头部

连接操作的第一步是在TCP模块处创建表示连接控制信息的头部
通过TCP头部中的发送方和接收方端口号可以找到要连接的套接字

收发数据：

协议栈收到数据会存放发送缓冲区，网络包尽量满再进行发送

网络包容纳数据长度：

- MTU：网络包最大长度，以太网中一般1500字节，为包含头部的总长度
- MSS：数据长度（MTU-头部）

TCP协议头部ACK号，进行丢包核验，若有包没收到则直接重传，只有此步进行错误补偿

根据ACK返回快慢动态调整等待时间

利用滑动窗口，不用等待ACK号返回，直接发后续包