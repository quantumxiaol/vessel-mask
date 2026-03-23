# 脑血管及当前模型分割能力说明

本文档对大脑的主要血管系统进行了梳理，并明确了当前使用的模型（TopCoW）所能分割的血管范围。

## 一、大脑的血管有哪些？

脑血管总体上分为两大系统：**动脉系统**和**静脉/静脉窦系统**。

### 1. 动脉系统
脑动脉主要分为**前循环**和**后循环**，负责为大脑的不同区域供血：
*   **前循环**：主要来自双侧颈内动脉（ICA）。
*   **后循环**：主要来自双侧椎动脉（VA）汇合成的基底动脉（BA）。
*   **Willis 环（Circle of Willis）**：将前循环和后循环连接起来的核心动脉环。

临床上最常提到的**三对主要脑动脉**是：ACA（大脑前动脉）、MCA（大脑中动脉）和 PCA（大脑后动脉）。

若进一步细看，脑动脉系统通常可分为以下几个层级：
*   **第一层（供血主干）**：ICA（颈内）、VA（椎动脉）、BA（基底动脉）。
*   **第二层（Willis 环及其核心连接支）**：ACom（前交通）、双侧 ACA、双侧 PCom（后交通）、双侧 PCA。
*   **第三层（较大的远端分支）**：如 MCA 的更远端皮质支、ACA / PCA 的更远端分支等。
*   **第四层（穿支 / 细小分支）**：如豆纹动脉（Lenticulostriate Arteries, LSAs）、基底动脉桥脑穿支等。这些血管极细，在常规影像中更难稳定显示和分割。

### 2. 静脉 / 静脉窦系统
除了动脉，脑内还有一套完整的静脉系统。静脉系统并不与动脉一一同名同行，通常分为三大类：
*   **浅静脉系统**
*   **深静脉系统**
*   **硬脑膜静脉窦**：主要包括上矢状窦、下矢状窦、直窦、横窦、乙状窦、海绵窦、岩上窦等。

（全脑静脉系统最终都会回流入颈内静脉，在影像大范围显影时非常重要，但请注意：当前的 TopCoW 模型不做静脉分割）。

### 3. 补充说明：常规脑 CTA 的显影能力
*   **优势**：显示较大的颅内动脉主干和主要分支能力强，现代 CTA 提供亚毫米、各向同性的 3D 数据，非常有利于血管解剖显示。
*   **局限**：CTA 本质是对比剂通过时刻的“单时间点快照”，对血流动力学信息反映有限。常规 CTA 往往受骨质重叠、静脉叠加的影响，而且越往远端、越细的穿支小血管，越难以看清。

---

## 二、现在的模型（TopCoW）能分割出哪些血管？

当前使用的 **TopCoW 模型**其任务边界非常明确：**主要针对 Willis 环内的核心动脉段进行解剖分段**，而不是将全脑所有血管（包含细小分支或静脉）都提取出来。

### 1. 模型输入 (Input)
*   **影像类型**：脑部 **TOF-MRA**（飞行时间法磁共振血管成像）或 **CTA**（计算机断层扫描血管成像）。
*   **预处理要求**：无须进行任何图像预处理 (without any image preprocessing)。

### 2. 模型输出 (Output)
模型执行的是多类别分割（Multiclass segmentation），针对 Willis 环相关的核心动脉段，输出包含背景在内的 **14** 个分类标签（背景 + 13 类独立动脉分段）：

*   **`0` : background** (背景)
*   **`1` : BA** (Basilar Artery，基底动脉)
*   **`2` : R-PCA** (Right Posterior Cerebral Artery，右侧大脑后动脉)
*   **`3` : L-PCA** (Left Posterior Cerebral Artery，左侧大脑后动脉)
*   **`4` : R-ICA** (Right Internal Carotid Artery，右侧颈内动脉)
*   **`5` : R-MCA** (Right Middle Cerebral Artery，右侧大脑中动脉)
*   **`6` : L-ICA** (Left Internal Carotid Artery，左侧颈内动脉)
*   **`7` : L-MCA** (Left Middle Cerebral Artery，左侧大脑中动脉)
*   **`8` : R-Pcom** (Right Posterior Communicating Artery，右侧后交通动脉)
*   **`9` : L-Pcom** (Left Posterior Communicating Artery，左侧后交通动脉)
*   **`10`: Acom** (Anterior Communicating Artery，前交通动脉)
*   **`11`: R-ACA** (Right Anterior Cerebral Artery，右侧大脑前动脉)
*   **`12`: L-ACA** (Left Anterior Cerebral Artery，左侧大脑前动脉)
*   **`13`: 3rd-A2** (Third A2 segment of anterior cerebral artery，变异出现的第三大脑前动脉 A2 段)

> **总结**：当前模型只专注于上述对应的 **13 类核心动脉段骨架**（即主要的前后循环主干和 Willis 环构成支）。如果业务场景需要提取深部远端微小分支或静脉血管，则超出了这一模型的适用范围。