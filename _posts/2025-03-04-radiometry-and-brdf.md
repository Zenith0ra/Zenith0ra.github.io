---
title: 辐射度量学与BRDF基础
categories: [Computer Science and Technology, Computer Graphics]
tags: [radiometry, BRDF, material, rendering, physical simulation]
math: true
description: "本文详细介绍辐射度量学与双向反射分布函数(BRDF)的基础概念、数学模型和应用，探讨如何从物理角度准确描述和模拟材质表面对光的反射特性。"
---

## 引言

我们周围的大千世界充满了各种各样、具有不同外观特性的物质和材料。在计算机图形学中，一个核心的挑战是如何真实地模拟这些物体与光的交互，从而在屏幕上“绘制”或“渲染”（即从模型生成图像的过程）出逼真的图像。这引出了一个关键问题：我们如何形式化地描述光线照射到物体表面时的行为？特别是，光线是如何被表面反射的？

本文的主题正是为了回答这个问题，我们将深入探讨**双向反射分布函数（Bidirectional Reflectance Distribution Function, BRDF）**。BRDF 提供了一个基于物理的框架，用于描述光线如何从任意入射方向被物体表面反射到任意出射方向，它是现代真实感渲染技术的基石。

## 回忆：Phong 光照模型及其局限性

在深入 BRDF 之前，让我们简要回顾一下之前讨论过的 Phong 光照模型。Phong 模型是一种经典的经验模型，它将物体表面的光照分解为三个主要部分：

1.  **环境光 (Ambient Light)**：模拟间接光照，提供基础亮度。
2.  **漫反射光 (Diffuse Light)**：模拟光线照射到粗糙表面后向各个方向均匀散射的效果，其强度遵循兰伯特余弦定律。
3.  **镜面反射光 (Specular Light)**：模拟光线照射到光滑表面后在特定方向（反射方向）产生的镜面高光。

通过将这三者叠加 (`环境光 + 漫反射光 + 镜面反射光 = 最终颜色`)，Phong 模型能够生成具有一定真实感的图像，特别是在表现光滑物体的高光方面。

然而，Phong 模型存在一些固有的缺点。首先，它是一个**经验模型**，其公式主要基于观察和直觉，并非严格基于物理定律，因此其模拟结果有时会偏离真实世界的光照现象。其次，它通常只适用于处理**点光源和方向光源**，对于更复杂的区域光源或环境光照处理能力有限。

为了追求更高的真实感和物理准确性，我们需要引入更严谨的理论基础。这便是**辐射度量学（Radiometry）**发挥作用的地方。基于辐射度量学，我们可以定义 BRDF，从而更精确地描述光的反射行为。

## 本节课内容概览：BRDF 材质

本文将围绕 BRDF 展开，主要内容包括：

1.  **预备知识**：辐射度量学（Radiometry）的基本概念。
2.  **BRDF 简述**：定义、核心性质以及与渲染方程的关系。
3.  **BRDF 模型**：介绍几种常见的 BRDF 模型（经验模型、物理模型、数据驱动模型）。
4.  **BRDF 的度量与评价**：如何获取和评估真实材质的 BRDF 数据。
5.  **进阶话题**：超越 BRDF 的次表面散射（Subsurface Scattering）与 BSSRDF。

## 预备知识：辐射度量学（Radiometry）

辐射度量学是研究电磁辐射（包括可见光）能量度量的物理学分支。理解其基本概念对于掌握 BRDF至关重要。

### 球面坐标 (Spherical Coordinate)

在图形学中，光线的方向至关重要。相比笛卡尔坐标 `(x, y, z)`，使用球面坐标 `(r, θ, ϕ)` 描述方向通常更为方便。

*   `r`：向量的长度（到原点的距离）。
*   `θ`：**极角 (Polar Angle)**，向量与 `z` 轴正方向的夹角（通常范围 `[0, π]`）。
*   `ϕ`：**方位角 (Azimuthal Angle)**，向量在 `x-y` 平面上的投影与 `x` 轴正方向的夹角（通常范围 `[0, 2π)`）。

三维笛卡尔坐标 `(x, y, z)` 和球面坐标 `(r, θ, ϕ)` 的转换关系如下：

*   从笛卡尔到球面：
    *   $r = \sqrt{x^2 + y^2 + z^2}$
    *   $\theta = \operatorname{acos}(z/r)$
    *   $\phi = \operatorname{atan}(y/x)$
*   从球面到笛卡尔：
    *   $x = r \sin(\theta) \cos(\phi)$
    *   $y = r \sin(\theta) \sin(\phi)$
    *   $z = r \cos(\theta)$

### 立体角 (Solid Angle)

立体角是平面角（弧度）在三维空间中的推广，用于衡量一个观察点对一个区域所张的“视野大小”。它定义为以观察点为球心，半径为 `r` 的球面上，该区域所投影的面积 `A` 与半径平方 `r²` 之比。

*   立体角的符号通常是 `ω` (omega)，单位为球面度 (steradian, sr)。
*   其微分形式为：
    $$ d\omega = \frac{dA}{r^2} $$
*   在球面坐标系下，球面上的一个无穷小面积微元 `dA` 可以表示为 $dA = (r d\theta)(r \sin\theta d\phi) = r^2 \sin\theta d\theta d\phi$。因此，立体角的微分形式也可以写为：
    $$ d\omega = \sin\theta d\theta d\phi $$
*   整个球面的立体角为 `4π` 球面度。一个半球面的立体角为 `2π` 球面度。

### 投影面积 (Foreshortened Area)

当一个表面微元 `dA` 被从某个方向观察时，其可见的有效面积会因观察角度而改变。如果观察方向与表面法向量 `n` 的夹角为 `θ`，则该表面微元在该观察方向上的**投影面积**为：
$$ dA_{\perp} = dA \cos\theta $$
这个概念在计算光通量时非常重要，因为只有垂直于光线传播方向的面积才决定了能量的密度。

### 光能 (Radiant Energy)

光能 $Q$ 是指电磁辐射能量的总和，可以理解为一定区域或时间内光子能量的总量。其单位是焦耳 (Joule, J)。

### 光通量 (Radiant Flux)

光能是不断运动的。**光通量** $Φ$ (Phi) 描述的是单位时间内穿过某个截面或由某个光源发出的光能。它是功率的一种形式。
$$ \Phi = \frac{dQ}{dt} $$
光通量的单位是瓦特 (Watt, W)。

### 发光强度 (Intensity)

**发光强度** $I$ 描述一个点光源（或一个微小光源）在特定方向上单位立体角内发出的光通量。
$$ I = \frac{d\Phi}{d\omega} $$
发光强度的单位是瓦特每球面度 (W/sr)。

### 光亮度 (Radiance)

**光亮度** $L$ 是辐射度量学中极其重要的一个概念，尤其在渲染中。它描述了**沿某一方向传播的光线的“亮度”**。具体来说，它定义为单位投影面积 ($dA_{\perp} = dA \cos\theta$)、单位立体角 ($d\omega$) 上的光通量 $Φ$。
$$ L = \frac{d^2\Phi}{dA \cos\theta \, d\omega} $$
其中，`θ` 是光线方向与表面法线的夹角。光亮度的单位是瓦特每平方米每球面度 (W/(m²·sr))。

Radiance 可以看作是描述渲染中“一条光线”携带能量的基本物理量。它有一个重要的性质：在真空中（或均匀介质中）沿直线传播时，其值保持不变（不考虑能量损失）。这使得它成为光线追踪等算法的核心。

Radiance 也可以看作是单位投影面积上的发光强度：
$$ L = \frac{dI}{dA \cos\theta} $$
其中 $dI = d\Phi / d\omega$ 是该面元在特定方向上的发光强度。

### 辉度 (Irradiance)

**辉度** $E$ 描述的是到达物体表面单位面积上的总光通量（来自所有方向的入射光）。
$$ E = \frac{d\Phi}{dA} $$
辉度的单位是瓦特每平方米 (W/m²)。它衡量的是物体表面受到的光照强度。

### 辉度与光亮度的关系

物体表面某一点的辉度 $E$ 是所有入射到该点的光线的光亮度 $L_i(ω_i)$ 在该点产生的贡献的总和。这些入射光线来自覆盖该点的整个半球 `Ω`。每条入射光线的贡献需要考虑其方向与表面法线的夹角 `θ_i`（即 `cos(θ_i)` 因子，因为辉度是按实际面积计算，而光亮度是按投影面积定义）。

因此，辉度可以表示为入射光亮度在入射半球 `Ω` 上的积分：
$$ E = \int_{\Omega} L_i(\omega_i) \cos\theta_i \, d\omega_i $$
其中，$L_i(\omega_i)$ 是沿入射方向 $\omega_i$ 到达该点的光亮度，$\theta_i$ 是 $\omega_i$ 与表面法线的夹角。只有当 $\cos\theta_i > 0$ (即 $\theta_i < 90°$) 时，光线才会对表面的辉度产生贡献。

## BRDF 简述

掌握了辐射度量学的基本概念后，我们现在可以正式定义和探讨 BRDF。

### BRDF 的定义

**双向反射分布函数 (BRDF)**，通常表示为 $f_r(\omega_i \rightarrow \omega_r)$ 或 $f_r(p, \omega_i, \omega_r)$（包含表面点 `p`），描述了光线从入射方向 $\omega_i$ 到达物体表面一点后，被反射到出射方向 $\omega_r$ 的比例和分布特性。

更精确地说，BRDF 定义为：**出射方向 $\omega_r$ 上的反射光亮度 $dL_r(\omega_r)$ 与来自入射方向 $\omega_i$ 的入射辉度 $dE_i(\omega_i)$ 之比**。
$$ f_r(\omega_i \rightarrow \omega_r) = \frac{dL_r(\omega_r)}{dE_i(\omega_i)} $$
由于 $dE_i(\omega_i) = L_i(\omega_i) \cos\theta_i d\omega_i$，其中 $L_i(\omega_i)$ 是入射光亮度，$\theta_i$ 是入射角，$d\omega_i$ 是入射方向的立体角微元，BRDF 也可以定义为出射光亮度与入射光亮度乘以 $\cos\theta_i d\omega_i$ 的比值：
$$ f_r(\omega_i \rightarrow \omega_r) = \frac{dL_r(\omega_r)}{L_i(\omega_i) \cos\theta_i d\omega_i} $$
BRDF 的单位是球面度的倒数 (sr⁻¹)。它是一个描述表面材质光学属性的四维函数（两个入射角 $(\theta_i, \phi_i)$ 和两个出射角 $(\theta_r, \phi_r)$）。

简而言之，BRDF 告诉我们，对于给定的入射光线，有多少能量会以怎样的角度分布被反射出去。它是绝大多数图形学算法中用于描述光反射现象的基本模型。

![BRDF-definition]({{ site.url }}/assets/img/2025-03-04-radiometry-and-brdf/BRDF-definition.png)

### BRDF 的性质

物理上有效的 BRDF 必须满足两个重要性质：

1.  **亥姆霍兹可逆性 (Helmholtz Reciprocity)**：
    交换入射光方向 $\omega_i$ 和出射光方向 $\omega_r$，BRDF 的值保持不变。这源于物理上的光路可逆原理。
    $$ f_r(\omega_i \rightarrow \omega_r) = f_r(\omega_r \rightarrow \omega_i) $$
    值得注意的是，我们之前提到的 Phong 模型（在其标准形式下）并不满足这个性质。

2.  **能量守恒 (Energy Conservation)**：
    物体表面反射的总能量不能超过入射的总能量。这意味着，对于任意给定的入射方向 $\omega_i$，所有出射方向 $\omega_r$ 的反射光能量总和必须小于或等于入射光能量。
    数学上，这表现为 BRDF 在整个出射半球 `Ω` 上的积分必须小于等于 1：
    $$ \int_{\Omega} f_r(\omega_i \rightarrow \omega_r) \cos\theta_r \, d\omega_r \le 1 \quad \forall \omega_i $$
    其中 $\theta_r$ 是出射角。如果等号成立，表示所有入射光都被反射（没有吸收或透射）。如果小于 1，则表示有部分能量被表面吸收或透射进入物体内部。

### 反射方程 (The Reflection Equation)

BRDF 的核心用途是计算给定表面点 `p` 在特定出射方向 $\omega_r$ 上的总反射光亮度 $L_r(p, \omega_r)$。这需要考虑所有可能的入射方向 $\omega_i$ 对该出射方向的贡献。根据 BRDF 的定义，来自特定入射方向 $\omega_i$ 的入射光 $L_i(p, \omega_i)$ 对出射方向 $\omega_r$ 的贡献是 $f_r(p, \omega_i \rightarrow \omega_r) L_i(p, \omega_i) \cos\theta_i d\omega_i$。

将所有入射方向（覆盖整个半球 `Ω`）的贡献积分起来，就得到了**反射方程**:
$$ L_r(p, \omega_r) = \int_{\Omega} f_r(p, \omega_i \rightarrow \omega_r) L_i(p, \omega_i) \cos\theta_i \, d\omega_i $$
在渲染中，我们最终看到的像素颜色，本质上就是由场景中对应着色点（shading point）沿相机方向 $\omega_r$ 反射的光线的 Radiance $L_r$ 决定的。

### 渲染方程 (The Rendering Equation)

反射方程描述了光线如何从表面反射。然而，物体本身也可能发光（例如灯泡或自发光材质）。为了完整描述表面一点沿特定方向出射的总光亮度 $L_o(p, \omega_o)$（这里用 `o` 代表 outgoing），我们需要在反射方程的基础上加上表面自身的发射项 $L_e(p, \omega_o)$（emitted radiance）。这就得到了著名的**渲染方程 (The Rendering Equation)**：

$$ L_o(p, \omega_o) = L_e(p, \omega_o) + \int_{\Omega} f_r(p, \omega_i \rightarrow \omega_o) L_i(p, \omega_i) \cos\theta_i \, d\omega_i $$

这里需要注意，入射光亮度 $L_i(p, \omega_i)$ 本身可能来自于场景中其他物体的出射光亮度 $L_o(p', \omega_i')$，其中 `p'` 是从 `p` 点沿 $-\omega_i$ 方向看到的另一点。这意味着渲染方程是一个**无限递归**的方程，光线会在场景中不断弹射。

渲染的本质问题，实际上就是求解这个积分方程。直接求解通常非常困难，因此在实践中我们使用各种数值方法来近似求解，例如下一讲将要讨论的**光线追踪 (Ray Tracing)**，它本质上是使用蒙特卡洛方法来解这个积分方程。

> **渲染方程之父：Jim Kajiya**
>
> 渲染方程由 Jim Kajiya 在他 1986 年于 SIGGRAPH 发表的里程碑式论文 "The Rendering Equation" 中首次提出。这篇论文奠定了现代物理渲染的基础。Kajiya 在计算机图形学领域，尤其是在渲染和硬件设计方面做出了卓越贡献，并因此获得了著名的 Steven Anson Coons 奖。
>
> **小插曲：SIGGRAPH 审稿**
>
> 1993年，Jim Kajiya 担任 SIGGRAPH '93 的技术程序主席，他对审稿流程进行了重大改革，这些改革沿用至今。他还写了一篇幽默而深刻的文章 "How to get your SIGGRAPH Paper rejected."，至今仍被图形学研究者们津津乐道。这体现了社区对严谨性和创新性的追求。

## BRDF 模型

直接使用四维的 BRDF 函数 $f_r(\omega_i, \omega_r)$ 进行计算可能非常复杂和耗时。为了方便和高效地在渲染中使用，研究者们提出了各种参数化的 **BRDF 模型**来近似真实材质的反射特性。这些模型大致可以分为三类：

1.  **经验模型 (Empirical Models)**
2.  **基于物理的模型 (Physical-based Models)**
3.  **数据驱动的模型 (Data-driven Models)**

### 经验模型 (Empirical Models)

经验模型主要基于对光照现象的观察和直觉，使用简洁的数学公式来快速估算反射效果。

*   **优点**：计算简单、速度快，易于实现，常用于实时渲染（如游戏）。
*   **缺点**：通常不严格遵循物理定律（如能量守恒、可逆性），参数缺乏明确的物理意义，模拟效果可能不够真实，尤其对于复杂材质。

#### 经验模型 1：Lambertian (理想漫反射)

Lambertian 模型是最简单的 BRDF 模型，它假设入射光线被表面均匀地反射到所有方向。

*   BRDF 是一个常数：$f_r(\omega_i \rightarrow \omega_r) = \frac{\rho_d}{\pi}$
*   $\rho_d$ (rho_d) 是**漫反射率 (Diffuse Albedo)**，表示表面反射的总能量与入射总能量之比（对于纯 Lambertian 表面，$\rho = \pi \times (\rho_d / \pi) = \rho_d$)。$\rho_d$ 的取值范围是 `[0, 1]`。
*   $1/\pi$ 的因子来自于能量守恒的归一化（常数 BRDF 在半球上的积分 `∫(ρ_d/π) cosθ dω = ρ_d`）。
*   出射光亮度 $L_r$ 只与总的入射辉度 $E_i$ 有关：$L_r = \int_{\Omega} \frac{\rho_d}{\pi} L_i(\omega_i) \cos\theta_i d\omega_i = \frac{\rho_d}{\pi} E_i$。

Lambertian 模型能很好地模拟理想的粗糙无光泽表面，如粉笔、未上釉的陶瓷、粗糙的墙面等。但它无法表现任何镜面反射（高光）效果，因此对于金属、塑料等光滑材质的模拟效果不佳。由于其简洁性，它经常作为更复杂模型（如 Phong）的漫反射分量。

#### Phong 模型

Phong 模型在 Lambertian 模型的基础上增加了一个镜面反射项，用于模拟光滑表面的高光。

*   Phong BRDF (注意：这与原始 Phong 光照方程略有不同，是将其概念转化为 BRDF 形式的一种常见表达)：
    $$ f_r(l \rightarrow v) = \frac{\rho_d}{\pi} + \rho_s \frac{(\mathbf{r} \cdot \mathbf{v})^s}{\dots ?} $$
    这里的 `l` 是入射光方向，`v` 是观察方向，`r` 是 `l` 关于表面法线 `n` 的理想镜面反射方向。$\rho_d$ 是漫反射系数，$\rho_s$ 是镜面反射系数。`s` 是**高光指数 (Shininess)**，控制高光的大小和锐利度（`s` 越大，高光越小越亮）。
*   *注：将 Phong 光照直接转为满足 BRDF 定义（单位 sr⁻¹）和性质（可逆性、能量守恒）的形式比较复杂，且有多种变体。原始讲义中给出的形式 $f_r(l \rightarrow v) = \rho_d + \rho_s (\mathbf{r} \cdot \mathbf{v})^s$ 更像是光照方程的直接挪用，单位和归一化可能存在问题，并且 $\rho_d$ 通常应除以 $\pi$。但按讲义内容呈现如下：*
    $$ f_r(l \rightarrow v) = \rho_d + \rho_s (\max(0, \mathbf{r} \cdot \mathbf{v}))^s $$
    其中 $\mathbf{r} = 2(\mathbf{n} \cdot \mathbf{l})\mathbf{n} - \mathbf{l}$。
*   **不可逆性**：标准的 Phong 模型不满足亥姆霍兹可逆性，即 $f_r(l \rightarrow v) \neq f_r(v \rightarrow l)$。
*   **优点**：计算仍然相对简单，能够同时表现漫反射和镜面反射，效果比 Lambertian 好。
*   **缺点**：物理不准确，缺乏能量守恒保证，参数物理意义不明确。

尽管有这些缺点，Phong 模型及其变种因其高效性，在实时渲染领域仍被广泛使用。

#### Phong 模型的扩展

*   **Blinn-Phong 模型**：为了计算优化，引入**半程向量 (Halfway Vector)** $\mathbf{h} = (\mathbf{l} + \mathbf{v}) / ||\mathbf{l} + \mathbf{v}||$。用 $\mathbf{n} \cdot \mathbf{h}$ 替代 Phong 模型中的 $\mathbf{r} \cdot \mathbf{v}$。
    $$ f_r(l \rightarrow v) \approx \frac{\rho_d}{\pi} + \rho_s' (\max(0, \mathbf{n} \cdot \mathbf{h}))^{s'} $$
    （同样，常数项和归一化因子可能有不同形式）。Blinn-Phong 在某些情况下（特别是当视线接近掠射角时）能更好地匹配实验数据，并且计算 `h` 通常比计算 `r` 更快。
*   **快速 Phong 绘制 (Fast Phong Shading)**：利用泰勒展开或查找表等技术加速 $(\mathbf{r} \cdot \mathbf{v})^s$ 或 $(\mathbf{n} \cdot \mathbf{h})^{s'}$ 的指数计算。
*   **可逆的 Phong 模型 (Modified Phong Model)**：为了满足可逆性，有研究者提出了修改版的 Phong 模型。讲义中提到的修改方式似乎是直接使用 $f_r(l \rightarrow v) = \rho_d + \rho_s (\mathbf{r} \cdot \mathbf{v})^s$ 并声称其满足可逆性 $f_r(l \rightarrow v) = f_r(v \rightarrow l)$。这需要对原始模型做更具体的调整或重新解释才能成立，例如确保 `r` 和 `v` 的对称性或采用不同的公式结构。标准的 Phong 或 Blinn-Phong 通常需要特定修改才能保证可逆性和能量守恒。

### 基于物理的模型 (Physical-Based Models, PBR)

与经验模型不同，物理模型尝试从**表面微观结构 (Microgeometry)** 和**光学原理 (Optics)** 出发，建立更精确的反射方程。

*   **核心思想**：大多数真实表面在微观尺度下都不是完美光滑的，而是由大量微小的**微平面 (Microfacets)** 组成。这些微平面的朝向（法线）存在一定的统计分布，这决定了表面的**粗糙度 (Roughness)**。
*   **假设**：光线与这些微平面发生相互作用（通常假设为镜面反射或漫反射），并且需要考虑微平面间的遮挡和阴影效应。
*   **关键组成部分**：
    *   **微平面分布函数 (Microfacet Distribution Function, D)**：描述微平面法线方向的统计分布。常用的有 Beckmann, GGX (Trowbridge-Reitz) 等。该函数通常与表面粗糙度参数相关。
    *   **菲涅尔项 (Fresnel Term, F)**：描述光线在两种不同折射率介质界面（如空气和物体表面）发生反射和折射的比例。这个比例取决于入射角、光的偏振态以及材质的折射率。对于非导体（电介质）和导体（金属），菲涅尔效应表现不同。在掠射角（grazing angles, 入射角接近 90°）时，几乎所有材质的反射率都会显著增加。
        *   精确的菲涅尔公式比较复杂，涉及光的 S 和 P 偏振。在图形学中，常用 **Schlick 近似 (Schlick's Approximation)** 来简化计算：
            $$ F(\theta_i) \approx F_0 + (1 - F_0)(1 - \cos\theta_i)^5 $$
            其中 $F_0 = (\frac{n_1 - n_2}{n_1 + n_2})^2$ 是法向入射时的反射率（$n_1, n_2$ 是两种介质的折射率）。
    *   **几何衰减项 (Geometric Attenuation Factor, G)**：描述由于微平面间的相互遮挡（masking, 从视线方向看）和阴影（shadowing, 从光源方向看）导致部分微平面无法贡献反射的效应。这个项取决于入射、出射方向以及表面粗糙度。

#### Cook-Torrance 模型

Cook-Torrance 模型是图形学中最早也是最经典的基于微平面理论的物理 BRDF 模型之一。它结合了漫反射项（通常是 Lambertian）和基于微平面理论的镜面反射项。

*   BRDF 结构：$f_r = k_d f_d + k_s f_s$，其中 $f_d$ 是漫反射部分（如 $\rho_d / \pi$），$f_s$ 是镜面反射部分。$k_d, k_s$ 是系数（可能与菲涅尔项相关，以保证能量守恒）。
*   镜面反射项 $f_s$：
    $$ f_s(\omega_i, \omega_o) = \frac{D(\mathbf{h}) F(\omega_i, \mathbf{h}) G(\omega_i, \omega_o, \mathbf{h})}{4 (\mathbf{n} \cdot \omega_i) (\mathbf{n} \cdot \omega_o)} $$
    其中：
    *   $\mathbf{h}$ 是半程向量。
    *   $D(\mathbf{h})$ 是微平面法线分布函数（例如，使用 Beckmann 分布）。只有法线方向恰好是 `h` 的微平面才能将光从 $\omega_i$ 精确反射到 $\omega_o$。
        *   Beckmann 分布示例： $D(\mathbf{h}) = \frac{1}{\pi \alpha^2 \cos^4\beta} \exp\left(-\frac{\tan^2\beta}{\alpha^2}\right)$，其中 $\alpha$ 是表面粗糙度参数，$\beta$ 是 `n` 和 `h` 的夹角。
    *   $F(\omega_i, \mathbf{h})$ 是菲涅尔项，通常用入射光线 $\omega_i$ 与微平面法线 `h` 的夹角计算。
    *   $G(\omega_i, \omega_o, \mathbf{h})$ 是几何衰减项，考虑 $\omega_i$ 和 $\omega_o$ 方向上的遮挡/阴影。一个常用的形式是 Smith G 函数，它结合了遮挡和阴影两部分。
        *   讲义中给出的 G 定义：$G = \min\left(1, \frac{2(\mathbf{n} \cdot \mathbf{h})(\mathbf{n} \cdot \omega_o)}{\omega_o \cdot \mathbf{h}}, \frac{2(\mathbf{n} \cdot \mathbf{h})(\mathbf{n} \cdot \omega_i)}{\omega_o \cdot \mathbf{h}}\right)$ (*注：此处分母 `ω_o ⋅ h` 可能有误，通常 G 函数形式更复杂，依赖于 D 函数的选择*）。
    *   分母中的 $4 (\mathbf{n} \cdot \omega_i) (\mathbf{n} \cdot \omega_o)$ 是校正因子，与坐标系变换和投影面积有关。

Cook-Torrance 模型能够解释一些经验模型无法模拟的现象：

*   **离轴高光 (Off-specular reflection)**：对于粗糙表面，最亮的高光可能偏离理想镜面反射方向。
*   **逆反射 (Retroreflection)**：部分光线会沿着接近入射方向反向散射回来，这在观察满月或路标时可以看到（边缘和中心亮度相似）。

#### 各向同性 (Isotropic) vs 各向异性 (Anisotropic) BRDF

*   **各向同性 BRDF**：如果将入射和出射方向同时围绕表面法线 `n` 旋转，BRDF 的值保持不变。这通常发生在表面微结构在各个方向上统计均匀的情况下（如大多数塑料、磨砂玻璃）。此时，BRDF 只需要考虑入射极角 $\theta_i$、出射极角 $\theta_r$ 以及方位角之差 $\Delta\phi = \phi_r - \phi_i$，因此是一个**三维函数** $f_r(\theta_i, \theta_r, \Delta\phi)$。
*   **各向异性 BRDF**：如果围绕法线旋转时 BRDF 值发生改变，则为各向异性。这通常发生在表面具有方向性微结构的情况下，如拉丝金属、绸缎、木纹、头发等。此时 BRDF 依赖于四个角度变量 $(\theta_i, \phi_i, \theta_r, \phi_r)$。

#### Ward 模型

Phong 和 Cook-Torrance 的原始形式通常是各向同性的。Ward 模型是处理**各向异性**反射的一个较早的著名模型。

*   提出者：Gregory Ward (1992)。
*   核心思想：使用椭圆高斯分布来描述微平面法线的分布，允许在不同切线方向上具有不同的粗糙度（用 $\alpha_x, \alpha_y$ 表示）。
*   模型形式（包含漫反射和镜面反射项）：
    *   **各向同性 Ward**:
        $$ f_r = \frac{\rho_d}{\pi} + \frac{\rho_s}{4\pi\alpha^2\sqrt{(\mathbf{n}\cdot\mathbf{l})(\mathbf{n}\cdot\mathbf{v})}} \exp\left(-\frac{\tan^2\delta}{\alpha^2}\right) $$
        其中 $\delta$ 是 `n` 和 `h` 的夹角，$\alpha$ 是表面坡度标准差（粗糙度）。
    *   **各向异性 Ward**:
        $$ f_r = \frac{\rho_d}{\pi} + \frac{\rho_s}{4\pi\alpha_x\alpha_y\sqrt{(\mathbf{n}\cdot\mathbf{l})(\mathbf{n}\cdot\mathbf{v})}} \exp\left[- \tan^2\delta \left(\frac{\cos^2\phi_h}{\alpha_x^2} + \frac{\sin^2\phi_h}{\alpha_y^2}\right)\right] $$
        其中 $\phi_h$ 是半程向量 `h` 在切平面上的方位角，$\alpha_x, \alpha_y$ 是沿切平面 x, y 方向的粗糙度。
*   **特点**：虽然引入了各向异性，但 Ward 模型通常被认为更偏向经验模型，因为它省略了菲涅尔项和精确的几何衰减项，使用了简化的归一化因子。

![ward-model]({{ site.url }}/assets/img/2025-03-04-radiometry-and-brdf/ward-model.png)

#### 其他物理模型

*   **Oren-Nayar 模型 (1994)**：扩展了 Lambertian 模型，用于模拟更粗糙的漫反射表面（如月球表面、黏土）。它假设微平面本身是 Lambertian 反射体，并考虑了微平面间的遮挡和掩蔽效应，使得表面在掠射角下看起来更亮（与 Lambertian 不同）。
*   **Poulin-Fournier 模型 (1990)**：使用一组平行的微小圆柱体来模拟各向异性表面（如拉丝金属）。
![wave-optics-model]({{ site.url }}/assets/img/2025-03-04-radiometry-and-brdf/wave-optics-model.png){: width="972" height="589" .w-50 .right}
*   **波动光学模型 (Wave Optics)**：当表面微结构尺寸与光的波长相当时，需要考虑光的衍射、干涉等波动效应。例如 He et al. (1991) 和 Stam (1999) 的工作。这些模型物理上更精确，能模拟彩虹色光泽（iridescence，如 CD 表面），但计算非常复杂，应用受限。

### 数据驱动的模型 (Data-driven Models)

当参数化模型无法准确捕捉某些复杂材质（如丝绸、天鹅绒）的外观时，可以使用数据驱动的方法。

*   **核心思想**：直接测量真实材质样本在各种光照和观察方向下的 BRDF 值，将这些测量数据存储起来。
*   **过程**：
    1.  使用特定设备（见下一节）对材质进行大量采样，得到高维的 BRDF 数据点。
    2.  由于原始数据量巨大（四维输入，一维输出），通常使用**降维技术**（如 PCA、NMF）来找到一个低维流形，用少量基函数或参数来表示 BRDF 数据。
    3.  代表性工作：Matusik et al. (2003) "A Data-Driven Reflectance Model"。
*   **优点**：能够非常灵活地表示各种复杂材质，无需对材质属性做假设。
*   **缺点**：需要大量的测量数据和存储空间；通常需要插值来获取未测量方向的值，可能丢失高频细节（如锐利高光）；获取数据成本高。

### BRDF 模型对比总结

*   **经验模型**：计算快，实现简单，视觉效果尚可。物理不准确，参数意义模糊。
*   **物理模型**：基于科学原理，参数具有物理意义，能模拟更多真实世界的反射现象，效果更逼真。通常计算更复杂。
*   **数据驱动模型**：最灵活，能捕捉极其复杂的材质。需要大量数据和预处理，可能丢失细节，插值可能引入问题。

现代 PBR (Physically Based Rendering) 流程主要依赖于物理模型（尤其是基于微平面的模型，如 Cook-Torrance 的变种 GGX），并结合数据驱动方法来获取或拟合模型参数。

## BRDF 的度量与评价

要使用物理或数据驱动模型渲染真实材质，首先需要获取这些材质的 BRDF 数据。

*   **动机**：
    *   为未知反射属性的新材料建模，以生成高度真实感的渲染结果。
    *   **逆渲染 (Inverse Rendering)**：从图像中反推出场景属性（包括光照、几何和材质 BRDF）。
*   **核心任务**：测量不同入射光方向 $\omega_i$ 和出射光方向 $\omega_r$ 组合下的 BRDF 值。

### 度量设备 (Measurement Devices)

测量 BRDF 通常需要复杂的设备，能够精确控制光源位置、相机位置以及样本姿态。

1.  **Gonioreflectometer (测角反射计)**：最经典的设备类型。通常包含一个光源臂和一个探测器（相机）臂，两者可以围绕样本表面上的一个点独立旋转，覆盖入射和出射的半球空间。
    *   可以固定样本，移动光源和相机。
    *   可以固定光源，移动相机和样本。
    *   Ward (1992) 开发了一种使用半镀银半球和鱼眼相机的 Gonioreflectometer，可以一次性采集整个半球的出射光信息，只需移动光源。

2.  **Light Stage (光场舞台)**：由 Debevec et al. (2000) 开发，特别适用于采集人脸等复杂几何物体的反射属性（包括 BRDF 和后续会讲的 BSSRDF）。
    *   通常是一个布满可控光源（如 LED）的球形或圆顶结构。
    *   物体（或人）位于中心，从少数几个固定视角拍摄物体在不同光源（单个或组合）照射下的图像。
    *   通过分析像素亮度随光照方向的变化，可以推断出表面的反射属性。

### 现代 BRDF 度量方法（基于机器学习）

传统的 BRDF 测量非常耗时（需要采样数千甚至数百万个方向组合）。近年来，研究者们利用机器学习来显著减少测量工作量。

*   **思路**：用少量精心设计的**光照模式 (Lighting Patterns)**（即同时点亮多个光源的组合）进行测量，然后利用模型从这些少量测量结果中恢复出完整的 BRDF 信息或其关键参数。
*   **示例流程 (基于自编码器)**：
    1.  **Lumitexel 表示**：将一个表面点在大量（如 10240 个）独立光源照射下的亮度响应表示为一个高维向量（Lumitexel），这个向量隐式地编码了该点的 BRDF。
    2.  **光照模式设计**：使用自编码器 (Autoencoder) 学习一个低维（如 32 维）的潜在空间来表示 Lumitexel。编码器的权重可以被解释为一组优化的光照模式。
    3.  **测量**：实际测量时，只使用这 32 种光照模式进行照射，得到 32 个亮度值。
    4.  **解码/恢复**：使用预先训练好的解码器（通常是全连接网络 FC），从这 32 个测量值恢复出完整的 10240 维 Lumitexel 近似值。
    5.  **参数拟合**：可以选择一个参数化的 BRDF 模型（如常用的 GGX 模型 $f_r = \frac{D(\mathbf{h})F(\mathbf{h})G(\mathbf{l},\mathbf{v})}{4(\mathbf{n}\cdot\mathbf{l})(\mathbf{n}\cdot\mathbf{v})}$），然后根据恢复的 Lumitexel 数据回归出该模型的参数（如粗糙度、基色等）。
    6.  **渲染**：使用拟合得到的 BRDF 参数在新的光照条件下进行渲染。

这种方法大大提高了 BRDF 采集的效率。

## 进阶：次表面散射 (Subsurface Scattering) 与 BSSRDF

目前讨论的所有 BRDF 模型都基于一个共同的假设：光线与物体表面的交互只发生在表面无限薄的一层，光线要么被反射，要么被吸收，**不会进入物体内部再出来**。

然而，现实世界中许多材质并非如此，它们是**半透明 (Translucent)** 的。例如：

*   玉石、大理石
*   蜡烛、肥皂
*   牛奶、果汁
*   皮肤、树叶

对于这些材质，光线会进入物体内部，在内部经历多次散射事件，然后从表面**不同于入射点**的位置射出。仅使用 BRDF 无法模拟这种现象，会导致渲染结果看起来像塑料或油漆，缺乏柔和、通透的质感。

### BTDF (Bidirectional Transmittance Distribution Function)

为了描述光线进入物体内部的行为，需要引入**双向透射分布函数 (BTDF)**，它描述了光线从一个方向入射后，透射进入物体内部并在另一侧以某个方向射出的特性。

### BSSRDF (Bidirectional Surface Scattering Reflectance Distribution Function)

更通用的函数是**双向表面散射反射分布函数 (BSSRDF)**，它统一描述了光线在物体表面附近（包括表面反射和内部散射再出射）的所有散射行为。BSSRDF 是一个更复杂的八维函数 $S(x_i, \omega_i, x_o, \omega_o)$，它关联了入射点 $x_i$、入射方向 $\omega_i$、出射点 $x_o$ 和出射方向 $\omega_o$。

*   **BSSRDF = BRDF (表面反射) + BTDF (透射) + 次表面散射**
*   当入射点和出射点相同时 ($x_i = x_o$)，BSSRDF 就退化为 BRDF。

### 次表面散射 (Subsurface Scattering, SSS)

完全模拟 BSSRDF 的计算量极大，且测量 BSSRDF 非常困难。在实践中，常用**次表面散射 (SSS)** 技术来近似模拟光线在物体内部的散射效果。

*   **核心思想**：光线进入物体后，在内部随机游走（散射），能量逐渐衰减。最终从附近某点射出。这个过程可以用扩散理论来近似。
*   **实用模型**：Jensen et al. (2001) 在 SIGGRAPH 上发表的 "A Practical Model for Subsurface Light Transport" 是一个里程碑式的工作。它提出了一种基于**偶极子 (Dipole)** 或多极子扩散近似的方法，将复杂的内部散射问题简化为相对容易计算的扩散方程求解。该模型允许艺术家通过指定散射颜色、吸收颜色和散射距离等参数来控制材质的半透明外观。
*   **感兴趣的读者可以参阅**：[https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf](https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf)

次表面散射对于渲染皮肤、蜡、玉石等材质至关重要，能显著提升渲染的真实感。

## 今日人物：Marc Levoy

Marc Levoy 是计算机图形学领域的杰出学者和实践者，现为斯坦福大学教授和 Adobe 副总裁。他在多个领域做出了开创性贡献：

*   **体绘制 (Volume Rendering)** (1980s)：开创了直接体绘制的技术。
*   **三维扫描 (3D Laser Scanning)** (1990s)：领导了著名的 Digital Michelangelo Project，使用激光扫描技术高精度数字化米开朗基罗的雕塑。
*   **计算摄影 (Computational Photography)**：对 Google 的街景 (Street View) 项目、Google Glass 以及智能手机摄影技术（如 HDR+, 人像模式）有重要贡献。

Levoy 教授荣获多项大奖，包括 1996 年的 ACM SIGGRAPH 计算机图形学成就奖，并于 2007 年当选为 ACM Fellow。他的工作极大地推动了图形学和相关领域的发展。

## 总结

本文深入探讨了计算机图形学中用于描述材质表面反射特性的核心概念——**双向反射分布函数 (BRDF)**。我们从辐射度量学的基础知识出发，详细定义了 BRDF 及其关键性质（可逆性、能量守恒），并介绍了它在渲染方程中的核心作用。随后，我们分类讨论了常见的 BRDF 模型，包括简单的经验模型（Lambertian, Phong）、更精确的物理模型（Cook-Torrance, Ward）以及灵活的数据驱动模型。我们还了解了 BRDF 数据的测量方法，从传统的测角反射计到现代基于机器学习的高效技术。最后，我们认识到 BRDF 的局限性，并引入了更通用的 **BSSRDF** 和**次表面散射 (SSS)** 的概念，以处理玉石、皮肤等半透明材质的逼真渲染。

理解和运用 BRDF 及其相关模型是实现高质量、物理真实感渲染的关键一步。
