---
title: 计算思维的应用 —— 粉碎后文件的复原问题
description: 本文探讨了粉碎后文件的复原问题，结合历史背景与现实需求，提出了通过计算思维提升古籍修复等行业工作效率的可能性。文章详细分析了碎片拼接的现有方法、子问题及其解决方案，包括碎片特征提取、边缘匹配度建模、优先级评估等，展示了计算机视觉与智能
categories: [Education and Guides, Coursework]
tags: [algorithm, document-reconstruction, computer-vision, image-processing, computational-thinking, tsinghua]
math: true
---

## 问题背景

- **历史背景**：1995年，德国政府委托一组档案专家还原遭撕毁的“斯塔西”文件。在过去8年时间里，包括数名前“斯塔西”雇员在内的15名工作人员一直在用镊子、胶带等工具试图拼凑这些碎纸片。但是工作进度极其缓慢，他们平均每天只能拼凑10份文件，从1995年至今的8年时间里，他们总共只拼凑了约250袋、共计50万张碎片。目前已经有四家德国公司竞争投标，试图揽下“斯塔西”文件的恢复工程。在这些公司中，有的把精力放在钻研碎纸的图案、颜色和穿孔，有的则打算使用以语义为基础的系统来寻找关键词汇和相关的内容。
- **现实需求**：目前，古籍修复是一项专业性强、高度依赖经验、效率低下的工作，有大量蕴藏在未修复古籍中的信息正在因老化损坏而永久丢失。以辽宁为例，全省目前有约30家古籍收藏单位，仅沈阳、大连两地的古籍收藏单位有修复人员，其他收藏单位都无力开展修复工作。辽宁省图书馆是国家首批重点古籍保护单位，馆藏古籍文献61万余册，其中未整理古籍15万余册。辽宁省图书馆古籍文献中心专家刘冰表示，相比偌大的馆藏，图书馆能实际参与古籍修复的人员仅有8人，古籍修复速度远远赶不上古籍折损、老化的速度。一个较为熟练的专业古籍修复师最多一年可以修100册，由此可以计算，辽宁省图书馆未整理古籍15万余册如果全部要完成修复的话，需要百余年，且前提是其他古籍不再继续遭到破坏。

**我们希望可以找到一种算法，可以以较高的效率从碎片中恢复出原始文件尽可能多的信息，以此提升古籍修复等行业的工作效率。**

- **相关案例**：2011年10月29日，DARPA组织了一场碎纸复原挑战赛，旨在寻找到高效的算法，对碎纸机处理后的碎纸屑进行复原。全美9000支队伍参与角逐，最终由来自旧金山的三名程序员组成的名为“All Your Shreds Are Belong To U.S.”参赛队伍获胜。其解决方案是：基于计算机视觉、图形学、纸张碎片的边缘性状等特征，设计出自动搜寻到最可能匹配的纸张碎片的仿真算法，同时对可能的碎片对进行人工筛选。在国内也有类似的比赛，如2013高教社杯全国大学生数学建模竞赛B题。

## 现有方法

- **非计算方法**：通过人工观察碎片的材质、颜色、图案、穿孔等特征，对碎片进行初步的分类，然后依靠情报人员的经验，使用镊子、胶带等工具将其拼凑到一起。
- **计算方法**：使用 SIFT、SURF 或 ORB 算法提取每个碎片的关键特征点，通过最近邻搜索与 RANSAC 算法进行匹配并消除错误；应用匈牙利算法、最短路径算法或遗传算法解决拼接问题。利用卷积神经网络、图神经网络、对抗神经网络等预测拼接关系。

## 子问题

### 1. 规整碎片的拼接（如碎纸机损毁）档案的拼凑与一般碎片的还原（如手撕碎片）有什么不同？

**碎片维度**（单面、双面、立体）、**碎片形状**（矩形、不规则多边形、曲线多边形、曲面）、**碎片材质**（纸、陶瓷、金属）都会影响拼接结果与效率。对不同的碎片进行初步分类，根据其类型提取不同的关键特征，并缩小比对范围，都有助于在后续信息提取、建模比对过程中减少计算量。

- **单/双面特征**：在提取碎片图案信息用于比较时，单面文件只需要提取单面信息，而双面文件不仅需要同时进行两次比较，若两面信息匹配结果不同，可能还需要引入算法以确定最佳结果。

- **形状特征**：在直线切割等碎片较为规整的情况下，提取边缘几何形状进行曲率特征匹配，或者提取边缘像素的颜色分布和纹理特性以计算相似性是较为有效的方法，然而在碎片边缘较复杂甚至受损时这种方法准确度有限。基于图像处理的拼接算法虽然可以克服该缺点，但面对体型较小、图像特征不明显的碎片时效果不理想。基于图论或深度学习的方法可以处理关系复杂的大量碎片，但是这两种方法计算复杂度高、算力需求大、耗时相对较长，现实中可能缺少硬件基础。

- **材质特征**：由于材料力学性能的不同，一些材料在特定作用下产生的碎片会具有某些特征。例如，冲击作用下破碎的陶瓷倾向以三重连接的方式断裂，形成“T”形或“Y”形的裂纹（图a,b）；此外大多数情况下碎片至少有一端以角-角的形式产生，而非图e,f所示。在比较碎片边缘信息时，将以上情况加以考虑，修改优先级，可以大幅降低时间复杂度，提高匹配效率。

- **内容特征**：碎片：原先属于同一部份的碎片更可能具有相似的内容，可以提取某些图案的分布模式，粗略估计碎片匹配的可能性。例如，包含标准汉字的图像的行均灰度值呈现出明显的等距变化，可以借此修正边缘匹配可能存在的错误；英文字母的行均灰度值存在跳变，相对汉字更不规律。

### 2. 如何将碎片（特别是边缘信息）录入为可量化指标以便计算机进行碎片间拼接的比较评估？

- **纵切碎片的基本特征**：形状、大小相同的矩形
  - **仅纵、横切**：至少两边为原始边缘，只需拼接另外两侧
  - **同时纵切和横切**：需要考虑各个方向的拼接可能
- **可利用的信息**：碎片大小、颜色（不适用于黑白文档）、笔迹和墨水特征（不适用于印刷文本）、轮廓信息（边缘形状的几何特征）、碎片边缘的图像像素信息
- **拼接思路**
  - **仅纵、横切情况**：提取边缘像素信息、计算两两边缘匹配度（定义边缘间的"距离"）、找到总"距离"最小的排列
  - **同时纵切和横切情况**：行、列边缘可分辨时，先在一个方向拼接，转化为仅纵、横切情形；不可分辨时，计算两两边缘距离，寻找总距离最小的拼接方案
  - **双面文件情形**：增加双面边缘距离的对比环节

> 注：后续讨论仅针对纵切情形

### 距离模型建立

#### 信息数据化——灰度矩阵（以黑白文档为例）

在黑白图像中，我们用灰度表示某一点颜色的深度，范围一般从0到255，白色为255，黑色为0。所以对于一个 $ m \times n $ 像素的图像，可转化为灰度矩阵 $ H $ ，像素点灰度值为 $ H_{ij} $ 。

将矩阵H的第一列与最后一列提取出来，得到向量：左边缘向量： $ L = (H_{11}, \cdots, H_{m1}) $ ，右边缘向量： $ R = (H_{1n}, \cdots, H_{mn}) $ 。

#### 边缘匹配度建模（边缘距离建模）

可以利用欧式距离定义任意两个边缘向量间的距离： $ d(\vec{x}, \vec{y}) = \sqrt{\sum_{i=1}^{m} (x_i - y_i)^2} $ 。对于碎片上下可区分的情况，只需考虑右边缘与其它碎片左边缘的距离，否则需要考虑两两边缘间的距离。

### 基于边缘距离进行拼接建模 —— TSP问题

**考虑碎片上下可分的情况**：碎片 1,2 的有向距离 $ 1 \rightarrow 2 $ 定义为 $ d(R_1, L_2) $ ，即将碎片2左边缘拼接上碎片1右边缘时的距离。将每一个碎片表示为平面上的点，建立所有的有向连接。有向箭头表示起始碎片右侧与目标碎片左侧拼接，大小为它们间的有向距离。整拼接即寻找遍历所有点的连线，寻找最优拼接即使得总距离最小。此时问题转换为**旅行商问题**：

假设有一个旅行商人要拜访N个城市，他必须选择所要走的路径，路径的限制是每个城市只能拜访一次，而且最后要回到原来出发的城市。路径的选择目标是要求得的路径路程为所有路径之中的最小值。对于本问题而言，区别在于两点间距离并不一致，箭头反向后将带来不同的距离。旅行商问题无法转换为算法复杂度为多项式的问题，是一个典型的NPC问题。

#### 近似有效算法建立 —— 贪心算法

**贪心算法（greedy algorithm）**：在对问题求解时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，算法得到的是在某种意义上的局部最优解。由于寻找总距离最小的算法复杂度极高，因此对于一个碎片，总是寻找与其距离最短的碎片加以拼接。

使用贪心算法时，由于不再进行穷举比较，因此需要选定一个起点。可以选择每一个碎片作为起点，分别拼接后比较每种拼接的总距离，也可以优先选择左侧边缘向量高（几乎全白）的作为起始碎片。

### 3. 如何通过碎片上的图案，判断其信息量，以此评估碎片的重要性？

碎片在原始文件中所含信息量各异，其重要性也因此而不同。例如，无字的空白碎纸片通常无需复原，因为它们缺乏实际信息价值。相反，带有文字的碎片则应被优先考虑复原，因其潜在的信息含量较高。进一步地，这些带字碎片的重要性还可以通过文字的内容、字体样式等视觉特征进行评估和比较，以此确定各碎片的优先级顺序。

对碎片优先级的评估确保了在有限的资源和时间条件下，能够最大限度地恢复文件的原貌，提高复原工作的整体效率。通过更加精准的资源分配和优先级排序，对于避免无效劳动、降低复原成本、加快工作进度具有重要意义。

#### 颜色和纹理分析

- 可以通过颜色直方图来表示每个碎片的颜色分布。确定使用的色彩空间，并对每一个颜色通道的色值进行统计。特定类型的文件可能会使用特定的颜色（如官方文件常用蓝色或红色水印）。颜色的使用可能与文档的重要性相关，如警告或注意事项通常使用醒目的颜色。也可以找到没有任何内容的碎片。
- 通过灰度共生矩阵计算图像中像素点间的灰度空间关系，描述其纹理特性。分析图像的对比度，对比度高的区域可能含有图像或文字的边缘，这是拼接碎片时的重要线索。也可以在文档中识别出用于安全目的的特殊图案或徽标，这些部分通常比较重要。

#### 信息熵的计算

- 首先使用OCR技术从碎片中提取文本，统计每个字符在文本中出现的频率。应用信息熵公式 $ H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i) $ ，其中 $ p(x_i) $ 是字符 $ x_i $ 的概率，这反映了文本的复杂度和信息内容。
- 信息熵的值越高，表明文本中的字符分布越均匀，信息的不确定性越高。在文档碎片复原中，高信息熵的文本可能包含更复杂或更丰富的信息，应当被优先考虑复原。

## 总结

碎纸复原技术揭示了信息恢复与传承的现代力量，尤其在古籍修复领域，传统方法的低效与信息丢失的危机迫切要求创新解决方案。从DARPA碎纸复原挑战赛到现代图像处理技术的发展，我们能够感受到计算机视觉、图形学与智能算法的巨大潜力，通过碎片分类、特征提取和优化拼接方法，可以显著提高复原效率，为古籍等珍贵文献的保护与修复开辟了崭新的方向。

在这一过程中，计算思维的引入，不仅帮助我们突破传统修复的瓶颈，还促使我们在面对复杂问题时，能够通过抽象与算法化的方式，寻找更为智能的解决路径。随着深度学习、大模型等技术的不断发展，碎片复原技术将愈加高效与精准，推动修复行业跨越时代，赋能文化遗产的保存与传承。