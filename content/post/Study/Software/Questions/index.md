---
title: 现代软件开发方法复习题
description: "Revision Questions of Contemporary software development"
slug: contemporary-software-dev-questions
date: 2024-07-07 00:00:00+0000
categories:
    - Software
tags:
    - study
    - Software
    - CS
weight: 1
---

# 现代软件开发方法复习题

## 一、概念解释

1. **Scrum**

**Scrum**是用于开发、交付和维持错综复杂产品 (complex products) 的敏捷框架 (framework)

以迭代（iterative）与增量（incremental）式的方式交付工作，每个迭代称作 Sprint

管理4-9人开发团队，高频率15分钟短会，让大家知道团队里每个成员在干什么

2. **基于计划-文档开发方法（Plan-and-Document based Development）**

软件的开发过程或生命周期依赖于预先仔细的规划、广泛而详尽的文档和精心的管 理使软件开发前景更加清晰。在编程之前，项目经理制定计划;在每个计划阶段，书写 详细的文档;根据计划制定项目的进度;项目的变更必须反应在文档中，可能的话在计 划中体现。著名的案例像瀑布模型、螺旋模型和 RUP 统一开发，这些都是根据基于计 划-文档开发方法实现的开发过程。

3. **DRY (Don’t Repeat Yourself) 无重复代码**

系统中的每一个知识(功能或特性)必须有单一的、无二义和明确的表示。敬告大家不要重复代码来达到复用的目的，一个规则只实现一次是面向对象编程中的基本准则， 旨在软件开发中减少重复的信息。

4. **MVC(软件作为服务的开发框架)**

MVC 全名是 Model View Controller，是模型(Model)-视图(View)-控制器(Controller) 的缩写，一种软件设计典范，用一种业务逻辑、数据、界面显示分离的方法组织代码， 将业务逻辑聚集到一个部件中，在改进和个性化定制界面及用户交互的同时，不需要重 新编写业务逻辑。MVC 被独特的发展起来用于映射传统的输入、处理和输出功能在一 个逻辑的图形化用户界面的结构中。

- Model(模型):模型与应用程序操作的数据有关，如何存储、操作以及改变数据。
- View(视图):呈现给用户的视图包含用户与之进行交互的模型信息。
- Controller(控制器):控制器是传递两个方向交互的中介。当用户与视图进行互动，一个特殊的控制器对此作出行动以回应用户的调用行为。

5. **SMART 用户故事**

用户故事，一个从人机交互(HCI)社区借来的方法，使非技术人员更易于提出功能需 求。SMART 缩写涵盖了用户故事令人满意的功能，判断一个用户故事好坏的标准: 

1. 确定性(Specific):功能的描述具体，不模糊; 
2. 可评估(Measurable):每一个合理的输入都有确定的预期结果; 
3. 可实现(Achievable):一个敏捷周期应当能实现一个用户故事，否则该故事的难度太 大， 应当进行分割; 
4. 相关性(Relevant):一个故事必须对一个或者多个涉众有商业价值;
5. 时间限制(Timeboxed):如果超出时间预算，需要停止开发对应的故事，此时要么放 弃，并将用户故事分割为更小的，要么重新评估，重新规划时间，安排剩下的时间和任务。

6. **TDD AND 红-绿-重构**

a. TDD(Test-driven development)测试驱动开发，是敏捷开发中的一项核心实践和技 术，也是一种设计方法论。TDD 的原理是在开发功能代码之前，先编写单元测试用例代 码，测试代码确定需要编写什么产品代码。

b. 测试驱动开发(TDD)循环又被称为红-绿-重构:

(1)在你写任何新的代码前，写一个这段代码行为应当有的在某个方面的测试。 由于被测试的代码还不存在，写下测试迫使你思考想要的代码如何表现，以及与可能存 在的协作者如何交互。我们把这种方法叫作“检测你想要的代码”。

(2)红色步骤:运行测试，并确认它失败了，因为你还没有实现能让它通过测试的 代码。

(3)绿色步骤:写出能使这个测试通过但不损坏其他已有的测试的尽可能简单的 代码。

(4)重构步骤:寻找重构代码或测试的机会，改变代码的结构来消除冗余、重复或 其他可能由添加新代码而引起的丑陋现象。测试能保证重构不会产生错误。

(5)重复以上步骤直到所有必须通过情景步骤的行为都完成。

7. **FIRST 测试**

创建一个好的测试有以下五个原则，首字母缩写为 FIRST:

(1)Fast(快速):它应该能简单和快速地运行与当前的编码任务相关的测试用例 来避免妨碍你连贯的思路。

(2)Independent(独立):任何测试都不应依赖于由别的测试引出的先决条件，这 样我们才能优先运行覆盖最近改动代码的测试子集。

(3)Repeatable(可重复):测试行为应当不依赖于外部因素。

(4)Self-checking(自校验):每一个测试都应该是能够自己决定测试是通过还是失败，不依赖于人工来检查它的输出。

(5)Timely(及时):测试应该在代码被测试的时候创建或更新。

8. **SOFA 代表四种特殊的味道**

1) 保持简短(short)，以便能迅速获取主要目的。  
    a) 在不同的地方有一些完全相同的代码  
    b) 提取方法将多余的代码装入，以便重复的地方都能调用

2) 只做一(one)件事，以便测试能集中于彻底检查这一件事。 
    a) 对一个类或方法的细小改动会波及到其他类或方法。  
    b) 通过移动方法和字段来把所有的数据集中到一个地方。

3) 输入少量(few)的参数，以便非常重要的参数值组合都能被测试到。  
    a) 多个数据项经常作为参数一起传递  
    b) 使用提取类或保存整个对象来创建一个类，将数据聚集，并传递该类的实例。

4) 维持一个固定级别的抽象(abstraction)，以便它不会在如何做和怎么做之间来回跳转。
    a) 一个类对另一个类实现的信息了解太多。  
    b) 如果这些方法确实在别处需要，那么可以移动方法或字段;如果两个类之间有重叠，那么就使用提取方法，或使用委托来隐藏实现细节。

9. **类间关系的 SOLID 原则  **

1) 单一责任原则。 一个类有且只有一个修改的理由
    a) 庞大的类，方法缺乏内聚性
    b) 提取类，移动方法
2) 开闭原则。对拓展保持开放，对修改保持封闭
    a) case 选择语句
    b) 使用策略或模板方法，可能结合其他抽象工厂模式;使用装饰器来避免子类数量 爆炸。
3) 里氏替换原则。替换一个类的子类应该保持正确的程序行为。  
    a) 拒绝继承，子类破坏性地覆盖一个继承的方法。
    b) 用委托来代替继承。
4) 依赖注入原则。实现在运行时可能会改变的协同类应该依赖一个中间状态的注入的 依赖。
    a) 单元测试需要特别设计的桩来创建嫁接;构造器硬编码到另一个构造器的调用， 而不是允许运行时决定使用哪个其他的类。
    b) 注入一个依赖到一个共享的接口，以分离类;根据需要使用适配器模式、外观模式、或者代理器模式来使跨变量的接口统一。  
5) 迪米特原则。仅和朋友交谈，把朋友的朋友当陌生人。
    a) 不合适的亲密
    b) 委托行为和调用委托方法。

10. **持续集成及开发**

持续集成是对部署的代码提高保障的一个关键技术，其中的每一个改变都会推送到 代码库中，引发一个集成测试集合来确保任何事都不会垮掉。持续集成通常被整合到一 个全面的开发过程中，而不是简单被动地进行测试。

CI 包括在部署之前运行一组集成测试，这通常比一个单独的开发者自己进行的集成 更广泛;CI 高度依赖自动化，工作流程通过当提交被推送到一个特定的代码库或者分支 时，自动触发 CI 进行测试。

11. **Free Software**

自由软件，根据自由软件基金会对其的定义，是一类可以不受限制地自由使用、复制、研究、修改和分发的，尊重用户自由的软件。这方面的不受限制正是自由软件最重要的本质，与自由软件相对的是专有软件，后者的定义与是否收取费用无关，事实上，自由软件不一定是免费软件，同时自由软件本身也并不抵制商业化。

二、选择题

DAABA
DBDDA
ADCBB
CDCAC
ABBAC
BBCDD
DCBAB
BADBC
ABDBA
BBD(B/D)A
DCBCA
CDDBC
CADCD
ADCAA
DAACC
ABDAD
DABCB
AAADB
ACBBA
(ABCDE)(ABCDE)CBD
ACDCA
ABCBC
CDCDA
AABBB
DACCD
BACDA
DACAD
ADDAA
BCBBB
BDADD
CBAC(iv i ii iii)D
CDCBA
ABB(AC)A
BCACA
DBADD
DBC

三、判断题

11010
10010
00110
10(?)11
11010

10101
10110
10100
(?1)(?1)(?0)11
11110

11111
10011
01(?0)11
01011
10101

11011
10000
10101
10(?1)11
1110(?1)

1(?0)110
01101
11011
00101
10010
10101
01