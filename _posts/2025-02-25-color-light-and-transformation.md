---
title: 计算机图形学中的颜色、光照与变换
categories: [Computer Science and Technology, Computer Graphics]
tags: [color-theory, lighting-model, transformation, RGB, Phong-model]
math: true
description: "本文详细介绍计算机图形学中的颜色视觉原理、光照模型与几何变换基础，探讨从人眼感知机制到颜色空间表示，以及Phong光照模型的数学原理与实现方法。"
---

## 1. 颜色视觉基础

### 1.1 什么是颜色

颜色是人眼对不同波长的光的能量的感知。不同波长的电磁波对应不同的颜色，人眼能感知到的光称为可见光，其波长范围在380nm到760nm之间。

>颜色不是物体的固有属性，而是光、物体和观察者三者相互作用的结果。同一物体在不同光源下可呈现不同颜色，而不同物体在特定光源下也可能呈现相同颜色。这种现象在色彩科学中称为"色彩恒常性"和"异谱同色"。
{:.prompt-info}

### 1.2 光的谱分布

"光"是由不同波长的电磁波按照某种能量分布混合叠加而成。例如，"白光"是由所有可见波长的电磁波以相等的强度混合得到的。

**光的谱分布**是指光在各个可见波长分量上的强度分布函数。与光类似，颜色也可以使用谱分布函数来进行描述。

人眼感知到的颜色由三个因素决定：
- 照明条件：光源发出的光的谱分布
- 物体材质：物体的反射光谱，描述物体对各个波长的光的吸收/反射能力
- 观察条件：光对人眼中视觉感受器的刺激

```python
# 模拟光的谱分布（Python代码示例）
import numpy as np
import matplotlib.pyplot as plt

# 可见光波长范围（380nm-760nm）
wavelengths = np.linspace(380, 760, 100)

# 模拟白光的谱分布（所有波长强度相等）
white_light = np.ones_like(wavelengths)

# 模拟日光的谱分布（近似值，实际更复杂）
daylight = 0.9 + 0.1 * np.sin((wavelengths - 570) * np.pi / 380)

# 模拟红色物体的反射谱
red_object_reflectance = 1.0 / (1 + np.exp(-0.1 * (wavelengths - 620)))

# 计算红色物体在白光下的反射光谱
reflected_spectrum = white_light * red_object_reflectance

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, white_light, label='白光谱分布')
plt.plot(wavelengths, red_object_reflectance, label='红色物体反射谱')
plt.plot(wavelengths, reflected_spectrum, label='反射光谱')
plt.xlabel('波长 (nm)')
plt.ylabel('相对强度')
plt.title('光谱分布示例')
plt.legend()
plt.grid(True)
```

>上述代码需要Python环境，以及NumPy和Matplotlib库支持
{:.prompt-info}

### 1.3 异谱同色现象

异谱同色现象指的是：观察条件不变，不同的物体材质仍然可能在某种特定的照明条件下被感知为同一种颜色。例如：紫光（400-450nm）的颜色可以由混合红光（620-750nm）和蓝光（450-495nm）得到。

这一现象是色彩科学和计算机图形学的重要基础，它使得我们可以用RGB三基色的混合来模拟自然界中的各种颜色。

## 2. 人眼结构与颜色感知

### 2.1 眼睛的基本结构

人眼的基本结构包括：
- 角膜：眼球最外层的透明组织
- 瞳孔：控制进入眼内光线量的开口
- 虹膜：围绕瞳孔的有色环状肌肉组织
- 晶状体：聚焦光线的透明结构
- 玻璃体：填充眼球后部的透明胶状物质
- 视网膜：包含感光细胞的眼内层
- 中央凹：视网膜上感光细胞最密集的区域
- 视神经：将视觉信息传递到大脑的神经

### 2.2 视网膜感光细胞

视网膜是人眼的感知器，包含三种主要的细胞类型：

**杆状细胞(Rod)**：
- 对亮度更敏感
- 在夜间视力中起主导作用
- 人眼中约有1.2亿个杆状细胞

**锥状细胞(Cone)**：
- 负责感知色彩
- 主要分为三类：对红光敏感(L型)、对绿光敏感(M型)、对蓝光敏感(S型)
- 人眼中约有500-600万个锥状细胞
- 主要集中在中央凹区域，因此人眼在视野中心处辨色力最强

**神经节细胞(Ganglion)**：
- 负责将感光细胞受到的刺激传递给视神经

>感光细胞分布特性：
>杆状细胞和锥状细胞在视网膜上的分布不均匀。锥状细胞主要集中在中央凹处，密度可达到每平方毫米175,000个；而杆状细胞在中央凹处很少，主要分布在周边区域，最大密度约为每平方毫米150,000个。
{:.prompt-info}

## 3. 颜色空间

### 3.1 颜色空间概述

颜色空间是以数学的方式描述、分类和重现颜色的系统。不同的应用场景可能使用不同的颜色空间。

常用的颜色空间包括：
- RGB：主要用于电子显示设备
- CMY/CMYK：主要用于印刷
- HSV/HSL：更符合人类直觉，便于颜色调整
- CIE XYZ：用于科学研究和标准化

### 3.2 RGB颜色空间

RGB (红绿蓝) 颜色空间在计算机图形学中使用最为广泛。在RGB颜色空间中：
- 一种颜色通过三通道向量 (r, g, b) 来表示
- 颜色表示为三个基本色：红色(R)，绿色(G)，蓝色(B)的线性组合：$C = rR + gG + bB$
- 在计算机中，通常将r, g, b三个分量分别规整化为[0, 1]内的浮点数；或[0, 255]内的8bit无符号整数

>为什么选择RGB作为基本颜色？
>RGB颜色空间的选择是基于人眼的生理结构。人眼中的锥状细胞分为三种，分别对红、绿、蓝三种颜色最敏感。通过混合这三种基本颜色，我们可以模拟人眼能够感知的大部分颜色。
{:.prompt-info}

```cpp
// RGB颜色表示与混合示例（C++）
struct Color {
    float r, g, b;  // 范围: [0, 1]
    
    Color(float r = 0, float g = 0, float b = 0) : r(r), g(g), b(b) {}
    
    // 颜色混合
    Color operator+(const Color& other) const {
        return Color(
            std::min(r + other.r, 1.0f),
            std::min(g + other.g, 1.0f),
            std::min(b + other.b, 1.0f)
        );
    }
    
    // 颜色缩放
    Color operator*(float scale) const {
        return Color(
            std::min(r * scale, 1.0f),
            std::min(g * scale, 1.0f),
            std::min(b * scale, 1.0f)
        );
    }
    
    // 转换为8位无符号整数表示 [0, 255]
    unsigned char toUChar(float value) const {
        return static_cast<unsigned char>(value * 255.0f + 0.5f);
    }
    
    // 获取颜色的8位RGB表示
    void getRGB(unsigned char& r_out, unsigned char& g_out, unsigned char& b_out) const {
        r_out = toUChar(r);
        g_out = toUChar(g);
        b_out = toUChar(b);
    }
};
```

>上述代码需要C++环境，包含标准库头文件 `<algorithm>` 以使用 `std::min` 函数
{:.prompt-info}

### 3.3 其他常用颜色空间

**CMY/CMYK颜色空间**：
- 主要用于印刷领域
- 基于减色混合原理，C（青）、M（洋红）、Y（黄）是三个基色
- K（黑）在CMYK中额外添加，用于增强打印黑色质量
- 与RGB的转换关系：
  $$\begin{pmatrix} C \\ M \\ Y \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} - \begin{pmatrix} R \\ G \\ B \end{pmatrix}$$

**HSV颜色空间**：
- H（色相）：表示颜色的种类，通常用0-360度表示
- S（饱和度）：表示颜色的纯度，范围为0-1或0%-100%
- V（明度）：表示颜色的亮度，范围为0-1或0%-100%
- 比RGB更符合人类对颜色的直觉认知

## 4. Phong光照模型

Phong光照模型是计算机图形学中最经典的局部光照模型之一，它将物体表面的光照效果分解为三个部分：环境光、漫反射光和镜面反射光。

### 4.1 光照模型的组成部分

**环境光（Ambient Light）**：
- 表示来自环境的均匀光照
- 计算公式：$I_a = k_a \cdot I_{ambient}$
- 其中$k_a$是材质的环境反射系数，$I_{ambient}$是环境光强度

**漫反射光（Diffuse Light）**：
- 基于Lambert余弦定律
- 表示光线均匀地向各个方向散射
- 计算公式：$I_d = k_d \cdot I_{light} \cdot \max(0, \vec{n} \cdot \vec{l})$
- 其中$k_d$是材质的漫反射系数，$I_{light}$是光源强度，$\vec{n}$是表面法向量，$\vec{l}$是指向光源的单位向量

**镜面反射光（Specular Light）**：
- 表示类似镜面的定向反射
- 计算公式：$I_s = k_s \cdot I_{light} \cdot \max(0, \vec{v} \cdot \vec{r})^{\alpha}$
- 其中$k_s$是材质的镜面反射系数，$\vec{v}$是观察方向，$\vec{r}$是反射方向，$\alpha$是光泽度（越大反射越集中）

### 4.2 完整的Phong光照模型

完整的Phong光照模型结合了上述三个部分：

$$I = I_a + I_d + I_s = k_a \cdot I_{ambient} + k_d \cdot I_{light} \cdot \max(0, \vec{n} \cdot \vec{l}) + k_s \cdot I_{light} \cdot \max(0, \vec{v} \cdot \vec{r})^{\alpha}$$

>Phong模型是一个经验模型，并不完全符合物理规律。在现代渲染系统中，通常使用更精确的基于物理的渲染（Physically Based Rendering, PBR）模型。然而，由于Phong模型简单且计算效率高，在实时渲染和教学中仍然广泛使用。
{:.prompt-info}

```glsl
// GLSL着色器中的Phong光照模型实现
// 顶点着色器输出
varying vec3 v_normal;      // 法向量
varying vec3 v_position;    // 顶点位置

// 片段着色器
void main() {
    // 材质属性
    vec3 materialAmbient = vec3(0.1, 0.1, 0.1);
    vec3 materialDiffuse = vec3(0.7, 0.7, 0.7);
    vec3 materialSpecular = vec3(1.0, 1.0, 1.0);
    float materialShininess = 32.0;
    
    // 光源属性
    vec3 lightPosition = vec3(10.0, 10.0, 10.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 ambientLight = vec3(0.3, 0.3, 0.3);
    
    // 计算各个向量
    vec3 normal = normalize(v_normal);
    vec3 lightDir = normalize(lightPosition - v_position);
    vec3 viewDir = normalize(-v_position);  // 假设在视图空间中
    vec3 reflectDir = reflect(-lightDir, normal);
    
    // 计算环境光分量
    vec3 ambient = materialAmbient * ambientLight;
    
    // 计算漫反射分量
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = materialDiffuse * lightColor * diff;
    
    // 计算镜面反射分量
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), materialShininess);
    vec3 specular = materialSpecular * lightColor * spec;
    
    // 合并所有光照分量
    vec3 result = ambient + diffuse + specular;
    
    // 输出最终颜色
    gl_FragColor = vec4(result, 1.0);
}
```

>上述代码是GLSL（OpenGL着色语言）着色器代码，需要在支持OpenGL/WebGL的图形系统中运行
{:.prompt-info}

## 5. 几何变换基础

### 5.1 坐标系与变换

计算机图形学中的几何变换涉及多个坐标系之间的转换：

- **模型坐标系（Model Space）**：物体自身的局部坐标系
- **世界坐标系（World Space）**：整个3D场景的全局坐标系
- **视图坐标系（View Space）**：以相机/观察者为中心的坐标系
- **裁剪坐标系（Clip Space）**：投影后用于裁剪的坐标系
- **屏幕坐标系（Screen Space）**：最终2D显示设备上的坐标系

### 5.2 基本变换类型

**平移变换（Translation）**：
- 沿着x、y、z轴移动物体
- 变换矩阵：
  $$T(t_x, t_y, t_z) = \begin{pmatrix} 
  1 & 0 & 0 & t_x \\
  0 & 1 & 0 & t_y \\
  0 & 0 & 1 & t_z \\
  0 & 0 & 0 & 1
  \end{pmatrix}$$

**缩放变换（Scaling）**：
- 调整物体在x、y、z轴方向上的大小
- 变换矩阵：
  $$S(s_x, s_y, s_z) = \begin{pmatrix} 
  s_x & 0 & 0 & 0 \\
  0 & s_y & 0 & 0 \\
  0 & 0 & s_z & 0 \\
  0 & 0 & 0 & 1
  \end{pmatrix}$$

**旋转变换（Rotation）**：
- 围绕某个轴旋转物体
- 例如，绕z轴旋转θ角度的矩阵：
  $$R_z(\theta) = \begin{pmatrix} 
  \cos\theta & -\sin\theta & 0 & 0 \\
  \sin\theta & \cos\theta & 0 & 0 \\
  0 & 0 & 1 & 0 \\
  0 & 0 & 0 & 1
  \end{pmatrix}$$

### 5.3 视图变换和投影变换

**视图变换（View Transformation）**：
- 将世界坐标系中的点转换到相机/视图坐标系
- 实际上是将相机放置到坐标原点，并调整其朝向

**投影变换（Projection Transformation）**：
- 将3D场景映射到2D平面上
- 主要包括透视投影和正交投影两种类型

透视投影矩阵（简化版，假设近平面z=n，远平面z=f）：

$$P = \begin{pmatrix} 
\frac{n}{r} & 0 & 0 & 0 \\
0 & \frac{n}{t} & 0 & 0 \\
0 & 0 & \frac{-(f+n)}{f-n} & \frac{-2fn}{f-n} \\
0 & 0 & -1 & 0
\end{pmatrix}$$

其中r和t分别表示近平面上视锥体的右边界和上边界。

```cpp
// C++中的变换矩阵实现
#include <cmath>

// 4x4矩阵类
class Matrix4 {
public:
    float m[4][4];
    
    // 单位矩阵
    static Matrix4 identity() {
        Matrix4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        return result;
    }
    
    // 平移矩阵
    static Matrix4 translation(float tx, float ty, float tz) {
        Matrix4 result = identity();
        result.m[0][3] = tx;
        result.m[1][3] = ty;
        result.m[2][3] = tz;
        return result;
    }
    
    // 缩放矩阵
    static Matrix4 scaling(float sx, float sy, float sz) {
        Matrix4 result = identity();
        result.m[0][0] = sx;
        result.m[1][1] = sy;
        result.m[2][2] = sz;
        return result;
    }
    
    // 绕Z轴旋转矩阵
    static Matrix4 rotationZ(float angleRadians) {
        Matrix4 result = identity();
        float c = cos(angleRadians);
        float s = sin(angleRadians);
        result.m[0][0] = c;
        result.m[0][1] = -s;
        result.m[1][0] = s;
        result.m[1][1] = c;
        return result;
    }
    
    // 矩阵乘法
    Matrix4 operator*(const Matrix4& other) const {
        Matrix4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }
};
```

>上述代码需要C++环境和数学库支持（<cmath>）
{:.prompt-info}
