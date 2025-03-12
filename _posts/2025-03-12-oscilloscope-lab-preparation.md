---
title: 数字逻辑实验一 示波器实验 预习报告
description: 数字逻辑实验一示波器实验的预习报告
categories: [Education and Guides, Coursework]
tags: [digital-logic, experiment, oscilloscope]
---

# 实验一 示波器实验

## 实验目的

1.  学习并熟悉示波器的使用方法
    
2.  学习如何使用内置信号发生器生成特定频率、特定种类的波形
    
3.  学习如何使用示波器光标进行信号参数的测量
    

## 实验内容

使用内置信号发生器或者时钟模块生成如下波形 / 接入对应波形，并使用示波器光标测量**高低电平的电压**以及**频率**，并将波形保存在 U 盘当中：

1.  使用内置信号发生器生成 100kHz 正弦波，占空比为 50％，直流电平为零，峰峰值 4V
    
    ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/100k_wave.png)
    
2.  使用内置信号发生器生成 1MHz TTL 方波，占空比为 50%
    
    ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/1M_wave.png)
    
3.  使用内置信号发生器生成 100Hz，0-5V 的三角波，占空比为 50%
    
    ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/100_wave.png)
    
4.  使用时钟模块生成测量实验模块 1MHz 输出
    
    ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/module_wave.png)
    

## 实验步骤

1.  探头校准
    
    +   打开示波器电源，将示波器探头连接到 `Demo 2` 端子上，并将探头的黑色夹子与示波器中间的接地端子连接，完成校准电路的搭建
    
    ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/Probe_Connect.png)
    
    +   按下 `Auto Scale` 键查看产生的波形，按下探头所连接的通道键`1` 和 `2`，然后按下屏幕下方`探头`下方的软键，查看屏幕下方`探头`处标识的衰减倍率是否与实际探头的衰减倍率一致；或者`Auto Scale` 后直接查看屏幕右侧的`通道`栏中的两个通道的倍率是否与实际探头的衰减倍率一致。
    
    ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/calibrate.png)
    
    +   在通道探头菜单中，按下 `无源探头检查` 下方的软键后选择 `OK` 进行检查。若为欠补偿 / 过补偿，请首先检查倍率是否匹配以及电路连接是否稳定，并重新进行探头检查。若仍无法通过测试，举手向在场助教或老师反馈情况，并使用专用工具调整探头上的微调电容进行探头校准。
2.  生成对应波形
    
    +   使用探头 1 或 2 连接信号发生器的输出或者时钟模块的 1M 输出
    
        - 内置信号发生器连接
        
        ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/wavegen_connect.png)

        - 时钟模块连接
        
        ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/module_wave_connect.png)
        
        > 为了让模块与示波器共地，可以增加一个带有 `GND` 插线孔的 IC14 模块，将探头的黑色夹子连接到这个插线孔，探头钩子连接到时钟模块的 `1M` 插线孔。
        {: .prompt-tip }
    
    +   使用示波器内置信号发生器时，按下示波器的 `Wave Gen` 按键，根据屏幕下面的软键进行设置生成需要的波形
    
        - 内置信号发生器 100kHz 正弦波
        
        ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/100k_wave_set.png)
        
        - 内置信号发生器 1MHz TTL方波

        ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/1M_wave_set.png)
        
        > TTL方波默认就是指0-5V的方波，可以通过设置 `幅度` `偏移` 来控制，还可以通过以下方式设置。
        > 这时信号发生器产生的信号就时0-5V的方波了。
        {: .prompt-tip }

        ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/1M_wave_set_tip.png)

        - 内置信号发生器 100Hz 0-5V的三角波
        
        ![](https://lab.cs.tsinghua.edu.cn/digital-logic-lab/doc/lab1/figure/100_wave_set1.png)
    
    +   按下示波器的 `Auto Scale` 按键自动调整显示，就可以看到稳定的输出信号波形。

3.  测量波形的相关参数
    
    - 手动测量
    
    推荐使用这个方式进行测量。
    
    按下示波器上的 `Cursors` 按钮，使用 `X1` `X2` `Y1` `Y2`四条光标进行测量，在屏幕右侧的 `光标` 栏中就可以看到数据结果。
    
    > 按下 `Cursors` 旋钮就可以快速切换光标。
    {: .prompt-tip }

    - 自动测量

    按下示波器上的 `Meas` 按钮，选择测量参数，在屏幕右侧的 `测量` 栏中就可以看到数据结果。
    
    > 如果波形信号不是很规整，自动测量的数据就会不准确，比如测量时钟模块的 1M 时钟的幅度时，由于波形的尖峰信号造成测量数据不正确。
    {: .prompt-warning }
    
4.  保存波形
    
    +   将 U 盘插入示波器
        
    +   按下 `Save/Recall` 按钮，根据屏幕下方的 `保存菜单` 下的软键可以设置保存的文件的格式、位置等，然后使用 `按下保存` 下方的软键将波形保存至 U 盘中。
        
    > 示波器可能只支持 8G 及以下的 U 盘，如果 U 盘无法识别，可以使用手机拍照。
    {: .prompt-tip }
    

## 实验报告要求

1.  4 个被测波形的波形图。
    
2.  这 4 个波形的测量频率，低电平，高电平值。
