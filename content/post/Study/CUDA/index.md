---
title: CUDA - 笔记
description: "Notes about CUDA"
slug: cuda-notes
date: 2024-05-11 00:00:00+0000
categories:
    - CUDA
tags:
    - study
    - AI
    - system
    - MLsys
    - CS
    - Notes
weight: 1
---

# CUDA

## Tutorial

参见[此篇教程](https://github.com/QINZHAOYU/CudaSteps/tree/master/)

### Helloworld

```C++
int main()
{
    // 主机代码
    // 核函数的调用
    // 主机代码

    return 0；
}
```

核函数与 c++ 函数的区别：

1. 必须加 `__global__` 限定；
2. 返回类型必须是空类型 `void`。

```C++
global void hell_from__gpu() { 
    // 核函数不支持 c++ 的 iostream。 
    printf("gpu: hello world!\n"); 
}
```    

调用核函数的方式：

```C++
hello_from_gpu<<<1, 1>>>
```

主机在调用一个核函数时，必须指明在设备中指派多少线程。核函数中的线程常组织为若干线程块：

1. 三括号中第一个数字是线程块的个数（number of thread block）；
2. 三括号中第二个数字是每个线程块中的线程数（number of thread in per block）。

一个核函数的全部线程块构成一个网格（grid），线程块的个数称为网格大小（grid size）。  
每个线程块中含有相同数目的线程，该数目称为线程块大小（block size）。

所以，核函数的总的线程数即 网格大小$*$线程块大小:

```C++
hello_from_gpu<<<grid size, block size>>>
```

调用核函数后，调用 CUDA 运行时 API 函数，同步主机和设备：

```C++
cudaDeviceSynchronize();
```

核函数中调用输出函数，输出流是先存放在缓冲区的，而缓冲区不会自动刷新。

#### CUDA线程组织

核函数的总线程数必须至少等于计算核心数时才有可能充分利用 GPU 的全部计算资源。

```C++
hello_from_gpu<<<2, 4>>>
```

网格大小是2，线程块大小是4，总线程数即8。核函数中代码的执行方式是 “单指令-多线程”，  
即每个线程执行同一串代码。

从开普勒架构开始，最大允许的线程块大小是 2^10 (1024)，最大允许的网格大小是 2^31 - 1（一维网格）。

线程总数可以由两个参数确定：

1. gridDim.x, 即网格大小；
2. blockDim.x, 即线程块大小；

每个线程的身份可以由两个参数确定：

1. blockIdx.x, 即一个线程在一个网格中的线程块索引，[0, gridDm.x);
2. threadIdx.x, 即一个线程在一个线程块中的线程索引，[0, blockDim.x);

网格和线程块都可以拓展为三维结构（各轴默认为 1）：

1. 三维网格 grid_size(gridDim.x, gridDim.y, gridDim.z);
2. 三维线程块 block_size(blockDim.x, blockDim.y, blockDim.z);

相应的，每个线程的身份参数：

1. 线程块ID (blockIdx.x, blockIdx.y, blockIdx.z);
2. 线程ID (threadIdx.x, threadIdx.y, threadIdx.z);

多维网格线程在线程块上的 ID；

```C++
tid = threadIdx.z * (blockDim.x * blockDim.y)  // 当前线程块上前面的所有线程数
    + threadIdx.y * (blockDim.x)               // 当前线程块上当前面上前面行的所有线程数
    + threadIdx.x                              // 当前线程块上当前面上当前行的线程数
```

多维网格线程块在网格上的 ID:

```C++
bid = blockIdx.z * (gridDim.x * gridDim.y)
    + blockIdx.y * (gridDim.x)
    + blockIdx.x
```

一个线程块中的线程还可以细分为不同的 **线程束（thread warp）**，即同一个线程块中  
相邻的 warp_size 个线程（一般为 32）。

对于从开普勒架构到图灵架构的 GPU，网格大小在 x, y, z 方向的最大允许值为 （2^31 - 1, 2^16 - 1, 2^16 -1）；  
线程块大小在 x, y, z 方向的最大允许值为 （1024， 1024， 64），同时要求一个线程块最多有 1024 个线程。

#### NVCC编译

nvcc 会先将全部源代码分离为 主机代码 和 设备代码；主机代码完整的支持 C++ 语法，而设备代码只部分支持C。

nvcc 会先将设备代码编译为 PTX（parrallel thread execution）伪汇编代码，再将其编译为二进制 cubin目标代码。  
在编译为 PTX 代码时，需要选项 `-arch=compute_XY` 指定一个虚拟架构的计算能力；在编译为 cubin 代码时，  
需要选项 `-code=sm_ZW` 指定一个真实架构的计算能力，以确定可执行文件能够使用的 GPU。

**真实架构的计算能力必须大于等于虚拟架构的计算能力**，例如：

```sh
-arch=compute_35  -code=sm_60  (right)
-arch=compute_60  -code=sm_35  (wrong)
```

#### CUDA程序框架

##### 单源文件 CUDA 程序基本框架

对于单源文件的 cuda 程序，基本框架为：

```
包含头文件

定义常量或宏

声明 c++ 自定义函数和 cuda 核函数的原型

int main()
{
    1. 分配主机和设备内存
    2. 初始化主机中数据
    3. 将某些数据从主机复制到设备
    4. 调用核函数在设备中计算
    5. 将某些数据从设备复制到主机
    6. 释放主机和设备内存
}

c++ 自定义函数和 cuda 核函数的定义
```

CUDA 核函数的要求：

1. 返回类型必须是 `void`，但是函数中可以使用 `return`（但不可以返回任何值）；
2. 必须使用限定符 `__glolbal__`，也可以加上 c++ 限定符；
3. 核函数支持 c++ 的重载机制；
4. 核函数不支持可变数量的参数列表，即参数个数必须确定；
5. 一般情况下，传给核函数的数组（指针）必须指向设备内存（“统一内存编程机制”除外）；
6. 核函数不可成为一个类的成员（一般以包装函数调用核函数，将包装函数定义为类成员）；
7. 在计算能力3.5之前，核函数之间不能相互调用；之后，通过“动态并行”机制可以调用；
8. 无论从主机调用还是从设备调用，核函数都在设备中执行（“<<<,>>>”指定执行配置）。

---

##### 自定义设备函数

核函数可以调用不带执行配置的自定义函数，即 **设备函数**。

设备函数在设备中执行、在设备中被调用；而核函数在设备中执行、在主机中被调用。

1. `__global__`修饰的函数称为核函数，一般由主机调用、在设备中执行；
2. `__device__`修饰的函数称为设备函数，只能被核函数或其他设备函数调用、在设备中执行；
3. `__host__`修饰主机段的普通 c++ 函数，在主机中被调用、在主机中执行，一般可以省略；
4. 可以同时用 `__host__` 和 `__device__` 修饰函数，从而减少代码冗余，此时编译器将 分别在主机和设备上编译该函数；
5. 不能同时用 `__global__` 和 `__device__` 修饰函数；
6. 不能同时用 `__global__` 和 `__host__` 修饰函数；
7. 可以通过 `__noinline__` 建议编译器不要将一个设备函数当作内联函数；
8. 可以通过 `__forceinline__` 建议编译器将一个设备函数当作内联函数。

设备函数可以有返回值。

#### 错误检测

##### Runtime

定义检查 cuda 运行时 API 返回值 `cudaError_t` 的宏函数。

```C++
#define CHECK(call)                                                     \
do {                                                                    \
    const cudaError_t error_code = call;                                \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA ERROR: \n");                                       \
        printf("    FILE: %s\n", __FILE__);                             \
        printf("    LINE: %d\n", __LINE__);                             \
        printf("    ERROR CODE: %d\n", error_code);                     \
        printf("    ERROR TEXT: %s\n", cudaGetErrorString(error_code)); \
        exit(1);                                                        \
    }                                                                   \
}while(0); 
```

因为核函数没有返回值，所以无法直接检查核函数错误。间接的方法是，在调用核函数后执行：

```
CHECK(cudaGetLastError());  // 捕捉同步前的最后一个错误。
CHECK(cudaDeviceSynchronize());  // 同步主机和设备。
```

##### Memory

CUDA 提供了 CUDA-MEMCHECK 的工具集，包括 memcheck, racecheck, initcheck, synccheck.

```
>> cuda-memcheck --tool memcheck [options] app-name [options]
```

对于 memcheck 工具，可以简化为：

```
>> cuda-memcheck [options] app-name [options]
```

#### 计时

```C++
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start)); // 创建cuda 事件对象。
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));  // 记录代表开始的事件。
cudaEventQuery(start);  // 强制刷新 cuda 执行流。

// run code.

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop)); // 强制同步，让主机等待cuda事件执行完毕。
float elapsed_time = 0;
CHECK(cudaEventElapsedTime(&curr_time, start, stop)); // 计算 start 和stop间的时间差（ms）。
printf("host memory malloc and copy: %f ms.\n", curr_time - elapsed_time);  
```

#### 影响GPT加速的关键因素

- 减少主机与设备之间的数据传输所花时间的占比
- 提高并行计算占比
- 核函数的并行规模

#### CUDA math library

[https://docs.nvidia.com/cuda/cuda-math-api/index.html](https://docs.nvidia.com/cuda/cuda-math-api/index.html)

### Memory

CUDA 中的内存类型有：全局内存、常量内存、纹理内存、寄存器、局部内存、共享内存。 CUDA 的内存，即设备内存，主机无法直接访问。