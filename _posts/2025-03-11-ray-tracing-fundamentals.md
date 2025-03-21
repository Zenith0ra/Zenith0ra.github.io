---
title: 光线跟踪算法基础
categories: [Computer Science and Technology, Computer Graphics]
tags: [ray-tracing, rendering, global-illumination, Whitted-algorithm, path-tracing]
math: true
description: "本文详细介绍光线跟踪算法的基本原理、历史发展与实现方法，包括光线投射、Whitted-Style光线跟踪和蒙特卡洛路径追踪，以及与传统光栅化渲染的区别。"
---

## 1. 光线跟踪概述

### 1.1 基本原理

光线跟踪是一种有效且被广泛使用的真实感绘制算法，它可以实现光栅化渲染算法难以处理的全局光照效果，通过模拟光线的多次弹射来创建逼真的图像。

>在物理世界中，我们之所以能看见物体，是因为光线从光源发出，经物体表面反复弹射后，部分光线最终进入视点、射入人们的眼中。光线跟踪算法正是基于这一物理过程进行模拟。
{:.prompt-info}

### 1.2 主要特征

光线跟踪的突出特征包括：

- 可以很容易地表现出阴影、反射、折射等引人入胜的视觉效果
- 适用于各种几何表示，包括基本几何形体（球体、立方体等）和复杂的物体表示方法（多边形网格、复合形体等）
- 作为一种经典的绘制方法，常用作ground truth，用于验证比较新的绘制方法的正确性

## 2. 光线跟踪的历史发展

### 2.1 里程碑

光线跟踪算法的发展史上最重要的里程碑是Turner Whitted于1980年首次提出的包含光反射和折射效果的模型：

- Turner Whitted, "An improved illumination model for shaded display", Communications of the ACM, Vol. 23, No. 6, 343-349, June 1980（SIGGRAPH 1979）

### 2.2 关键人物

Turner Whitted是光线跟踪算法的奠基人：

- 于1978年在北卡罗来纳州立大学获得Ph.D学位，之后加入贝尔实验室
- 一生共发表36篇论文，其中11篇发表在SIGGRAPH，2篇发表在Communication of the ACM
- 2003年当选美国工程院院士

## 3. 光线跟踪算法类型

### 3.1 光线投射

最基本的光线跟踪形式，仅考虑从视点到场景中第一个交点的光线。

### 3.2 Whitted-Style光线跟踪

递归式光线跟踪算法，考虑反射和折射：

```cpp
Color IntersectColor(Vector3 vBeginPoint, Vector3 vDirection) {
    // 确定交点
    Point IntersectPoint = FindIntersection(vBeginPoint, vDirection);
    Color result = ambient_color;
    
    // 计算每个光源的贡献
    for (each light in scene) {
        result += CalculateLocalShading(IntersectPoint, light);
    }
    
    // 递归计算反射和折射
    if (surface.isReflective()) {
        Vector3 reflectDir = CalculateReflection(vDirection, normal);
        result += reflectCoeff * IntersectColor(IntersectPoint, reflectDir);
    } 
    else if (surface.isRefractive()) {
        Vector3 refractDir = CalculateRefraction(vDirection, normal, ior);
        result += refractCoeff * IntersectColor(IntersectPoint, refractDir);
    }
    
    return result;
}
```

>上述代码是C++伪代码，实际实现中需要几何库来处理向量计算和光线-物体求交
{:.prompt-info}

### 3.3 蒙特卡洛光线跟踪（路径追踪）

基于概率的方法，通过随机采样来处理更复杂的光照场景，可模拟漫反射等更真实的效果。

## 4. 光线跟踪 vs 光栅化

### 4.1 光栅化

- 优点：速度快，有成熟渲染流水线，适合实时渲染（>30FPS），如游戏等
- 缺点：质量较低，难以模拟光线的多次弹射

### 4.2 光线跟踪

- 优点：质量更高，更接近真实物理世界
- 缺点：速度较慢，在RTX系列显卡问世前主要用于离线渲染，如电影等

>近年来，随着NVIDIA RTX等专用硬件的发展，实时光线跟踪已经开始应用于游戏和交互式应用中，如《控制》、《赛博朋克2077》等游戏都采用了实时光线跟踪技术。
{:.prompt-info}