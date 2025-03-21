---
title: 辐射度量学与BRDF基础
categories: [Computer Science and Technology, Computer Graphics]
tags: [radiometry, BRDF, material, rendering, physical simulation]
math: true
description: "本文详细介绍辐射度量学与双向反射分布函数(BRDF)的基础概念、数学模型和应用，探讨如何从物理角度准确描述和模拟材质表面对光的反射特性。"
---

## 1. 引言

在计算机图形学中，如何真实地绘制渲染（从模型生成图像的过程）以及如何形式化描述物体表面的材质属性，是实现真实感图像的关键挑战。本文将介绍双向反射分布函数（BRDF）——一个关于光线如何被物体表面反射的物理描述模型。

>BRDF相较于Phong等经验光照模型，更加符合物理规律，能够模拟更丰富多样的材质效果，是现代真实感渲染的基础。
{:.prompt-info}

## 2. 从Phong模型到BRDF

### 2.1 Phong模型回顾

Phong光照模型将物体的着色拆解为三个部分：
- 环境光（Ambient Light）
- 漫反射光（Diffuse Light）
- 镜面反射光（Specular Light）

表达式为：

$$I = k_a I_a + k_d I_d (\vec{N} \cdot \vec{L}) + k_s I_s (\vec{R} \cdot \vec{V})^n$$

其中：
- $k_a$, $k_d$, $k_s$ 分别是环境光、漫反射和镜面反射的系数
- $I_a$, $I_d$, $I_s$ 分别是环境光、漫反射和镜面反射的光强
- $\vec{N}$ 是表面法向量
- $\vec{L}$ 是指向光源的单位向量
- $\vec{R}$ 是反射向量
- $\vec{V}$ 是视线向量
- $n$ 是高光系数，控制镜面反射的锐度

### 2.2 Phong模型的局限性

Phong模型存在以下缺点：
- 作为经验模型，并不完全符合物理规律
- 仅支持点光源和方向光源
- 难以表现复杂材质的光学特性
- 不遵循能量守恒定律
- 双向反射特性不够准确

要更加贴合物理现实，我们需要：
- 引入物理学中的辐射度量学
- 使用BRDF来描述光的反射行为

## 3. 辐射度量学基础

### 3.1 球面坐标系统

光线主要通过方向来描述，通常使用球面坐标表达比笛卡尔坐标更为便捷。

球面坐标系中的方向向量由三个分量指定：
- $r$ 表示向量的长度
- $\theta$ 表示向量和z轴正方向的夹角
- $\phi$ 表示向量在x-y平面上的投影和x轴正方向的夹角

笛卡尔坐标$(x,y,z)$与球坐标$(r,\theta,\phi)$的转换关系：

$$
\begin{cases}
r = \sqrt{x^2 + y^2 + z^2} \\
\theta = \arccos(z/r) \\
\phi = \arctan(y/x)
\end{cases}
$$

反之：

$$
\begin{cases}
x = r\sin\theta\cos\phi \\
y = r\sin\theta\sin\phi \\
z = r\cos\theta
\end{cases}
$$

### 3.2 立体角

立体角描述了一个圆锥体在以其顶点为球心、半径为1的球面上所张成的视野大小，是平面角在三维空间的自然推广。

立体角的单位是球面度（steradian，简写为sr），最大值为全角$4\pi$，该最大值可以在区域为整个球面时取到。

立体角$\omega$的微分形式：

$$d\omega = \frac{dA}{r^2}$$

在球面坐标系下：

$$d\omega = \sin\theta d\theta d\phi$$

### 3.3 投影面积

投影面积描述了物体表面相对于某个视线方向下的可见面积。对于面积微元$dA$，其沿着与法向夹角为$\theta$方向的可见面积为：

$$dA_{\text{projected}} = dA\cos\theta$$

这一概念对于理解光的传播和辐射强度至关重要，被称为Lambert余弦定律。

### 3.4 辐射量与单位

辐射度量学定义了一系列精确描述光能传播的物理量：

1. **辐射能(Radiant Energy)**
   - 符号：$Q$
   - 单位：焦耳(J)
   - 定义：电磁辐射携带的能量总量

2. **辐射通量/功率(Radiant Flux/Power)**
   - 符号：$\Phi$
   - 单位：瓦特(W = J/s)
   - 定义：单位时间内通过表面的辐射能，$\Phi = \frac{dQ}{dt}$

3. **辐照度(Irradiance)**
   - 符号：$E$
   - 单位：瓦特/平方米(W/m²)
   - 定义：单位表面积接收的辐射通量，$E = \frac{d\Phi}{dA}$

4. **辐射强度(Radiant Intensity)**
   - 符号：$I$
   - 单位：瓦特/球面度(W/sr)
   - 定义：单位立体角内的辐射通量，$I = \frac{d\Phi}{d\omega}$

5. **辐亮度(Radiance)**
   - 符号：$L$
   - 单位：瓦特/球面度/平方米(W/sr/m²)
   - 定义：单位投影面积、单位立体角内的辐射通量，$L = \frac{d^2\Phi}{d\omega dA\cos\theta}$

辐亮度是渲染方程中最核心的概念，它描述了特定方向上的光能密度。

```python
# 计算球面上立体角的代码示例
import numpy as np

def solid_angle(theta_min, theta_max, phi_min, phi_max):
    """计算球面上一个区域的立体角
    
    参数:
    theta_min, theta_max: 极角的范围 (0到π)
    phi_min, phi_max: 方位角的范围 (0到2π)
    
    返回值:
    立体角（单位：球面度）
    """
    return (np.cos(theta_min) - np.cos(theta_max)) * (phi_max - phi_min)

# 计算漫反射表面的辐照度与辐亮度的关系
def diffuse_irradiance_to_radiance(irradiance):
    """计算完美漫反射表面的辐亮度
    
    对于完美漫反射表面，各个方向的辐亮度相同，且满足:
    L = E / π
    
    参数:
    irradiance: 表面接收的辐照度 (W/m²)
    
    返回值:
    辐亮度 (W/sr/m²)
    """
    return irradiance / np.pi
```

>上述代码需要NumPy库支持
{:.prompt-info}

## 4. BRDF定义与性质

### 4.1 BRDF数学定义

双向反射分布函数(Bidirectional Reflectance Distribution Function, BRDF)描述了入射光能量如何被表面反射到不同出射方向的比例。

BRDF的数学定义为：

$$f_r(\omega_i, \omega_o) = \frac{dL_o(\omega_o)}{dE_i(\omega_i)} = \frac{dL_o(\omega_o)}{L_i(\omega_i) \cos\theta_i d\omega_i}$$

其中：
- $\omega_i$ 是入射光方向
- $\omega_o$ 是出射光方向
- $L_i$ 是入射辐亮度
- $L_o$ 是出射辐亮度
- $E_i$ 是入射辐照度
- $\theta_i$ 是入射角（入射方向与法线的夹角）

BRDF的单位是1/sr (1/球面度)。

### 4.2 BRDF的物理性质

物理上合理的BRDF必须满足以下性质：

1. **非负性**：$f_r(\omega_i, \omega_o) \geq 0$
   - BRDF任何情况下都不能为负，否则会违反能量守恒

2. **亥姆霍兹互易性(Helmholtz Reciprocity)**：$f_r(\omega_i, \omega_o) = f_r(\omega_o, \omega_i)$
   - 光路可逆性，交换入射和出射方向，BRDF值不变

3. **能量守恒**：$\forall \omega_i, \int_{\Omega} f_r(\omega_i, \omega_o) \cos\theta_o d\omega_o \leq 1$
   - 反射能量不能超过入射能量，等式成立时为完全无损反射

4. **各向同性(可选)**：$f_r(\theta_i, \phi_i, \theta_o, \phi_o) = f_r(\theta_i, \theta_o, |\phi_i - \phi_o|)$
   - 对于许多材质，BRDF只与入射和出射方向的相对方位角有关

### 4.3 常见BRDF模型

#### 4.3.1 Lambert漫反射模型

最简单的BRDF模型，完美漫反射表面在所有方向上均匀反射光线：

$$f_r(\omega_i, \omega_o) = \frac{\rho_d}{\pi}$$

其中$\rho_d$是漫反射率（反照率），范围在[0,1]之间。

#### 4.3.2 Phong BRDF模型

将Phong光照模型转换为BRDF形式：

$$f_r(\omega_i, \omega_o) = \frac{k_d}{\pi} + k_s \frac{n+2}{2\pi} (\vec{R} \cdot \vec{V})^n$$

其中：
- $k_d$ 是漫反射系数
- $k_s$ 是镜面反射系数
- $n$ 是高光系数
- $\vec{R}$ 是反射向量
- $\vec{V}$ 是视线向量

#### 4.3.3 Blinn-Phong BRDF模型

Blinn-Phong模型是Phong模型的改进版本，计算更高效：

$$f_r(\omega_i, \omega_o) = \frac{k_d}{\pi} + k_s \frac{n+8}{8\pi} (\vec{N} \cdot \vec{H})^n$$

其中$\vec{H}$是半角向量，$\vec{H} = \frac{\vec{L} + \vec{V}}{|\vec{L} + \vec{V}|}$。

#### 4.3.4 Cook-Torrance微表面模型

更复杂但物理准确的模型，考虑了表面微观几何结构：

$$f_r(\omega_i, \omega_o) = \frac{k_d}{\pi} + \frac{DFG}{4(\omega_i \cdot \vec{N})(\omega_o \cdot \vec{N})}$$

其中：
- $D$ 是法线分布函数(NDF)，描述微表面法线分布
- $F$ 是菲涅尔项，描述光在不同角度的反射率变化
- $G$ 是几何项，考虑微表面间的遮挡和阴影

常用的法线分布函数包括：

1. **Beckmann分布**：
   $$D_{Beckmann}(\vec{H}) = \frac{1}{\pi \alpha^2 \cos^4 \theta_h} \exp\left(-\frac{\tan^2 \theta_h}{\alpha^2}\right)$$

2. **GGX分布**：
   $$D_{GGX}(\vec{H}) = \frac{\alpha^2}{\pi((\vec{N} \cdot \vec{H})^2 (\alpha^2-1) + 1)^2}$$

菲涅尔项通常使用Schlick近似：
$$F_{Schlick}(\cos\theta) = F_0 + (1 - F_0)(1 - \cos\theta)^5$$

其中$F_0$是垂直入射时的反射率。

## 5. 渲染方程与BRDF的应用

### 5.1 渲染方程

Kajiya在1986年提出的渲染方程是现代真实感渲染的基础：

$$L_o(p, \omega_o) = L_e(p, \omega_o) + \int_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (\omega_i \cdot \vec{N}) d\omega_i$$

其中：
- $L_o$ 是点$p$沿$\omega_o$方向的出射辐亮度
- $L_e$ 是点$p$自身发光的辐亮度
- $f_r$ 是BRDF
- $L_i$ 是点$p$接收到的来自$\omega_i$方向的入射辐亮度
- $\Omega$ 是上半球空间

渲染方程的物理含义是：出射辐亮度等于自发光辐亮度加上所有入射方向的光线经BRDF反射后的辐亮度总和。

### 5.2 BRDF在渲染中的应用

#### 5.2.1 基于物理的渲染(PBR)

PBR是现代游戏和电影行业的主流渲染方法，它使用物理上准确的BRDF模型，通常将材质参数化为：
- 漫反射率/基础色(Base Color)
- 金属度(Metalness)
- 粗糙度(Roughness)
- 环境光遮蔽(Ambient Occlusion)
- 法线贴图(Normal Map)

```glsl
// PBR渲染的GLSL片段着色器示例（简化版）
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

const float PI = 3.14159265359;

// 法线分布函数 (GGX/Trowbridge-Reitz)
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// 几何遮蔽函数 (Smith's Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// 组合几何项
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// 菲涅尔方程 (Schlick近似)
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

void main() {   
    // 材质属性
    vec3 albedo = texture(albedoMap, TexCoords).rgb;
    float metallic = texture(metallicMap, TexCoords).r;
    float roughness = texture(roughnessMap, TexCoords).r;
    float ao = texture(aoMap, TexCoords).r;
    
    // 输入照明数据
    vec3 N = getNormalFromMap();
    vec3 V = normalize(camPos - WorldPos);
    
    // 计算反射率F0
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);
    
    // 直接光照贡献
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) {
        vec3 L = normalize(lightPositions[i] - WorldPos);
        vec3 H = normalize(V + L);
        
        // 计算辐射度
        float distance = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;
        
        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);   
        float G = GeometrySmith(N, V, L, roughness);    
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
           
        vec3 nominator = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
        vec3 specular = nominator / max(denominator, 0.001);
        
        // 考虑能量守恒
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        float NdotL = max(dot(N, L), 0.0);        
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // 环境光照贡献
    vec3 ambient = calculateAmbientLighting(albedo, metallic, roughness, N, V);
    
    vec3 color = ambient + Lo;
    
    // HDR色调映射和伽马校正
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2)); 
    
    FragColor = vec4(color, 1.0);
}
```

>上述代码为GLSL片段着色器代码，需要在OpenGL/WebGL环境中使用
{:.prompt-info}

#### 5.2.2 BRDF的测量与捕获

现实材质的BRDF非常复杂，通常通过专用设备进行测量：
- 特制的机械臂和光源在不同角度捕获材质的反射特性
- 测量数据通常以四维表格形式存储
- Cornell University和Columbia University等机构提供了公开的BRDF数据库

#### 5.2.3 重要性采样

由于渲染方程中的积分难以解析求解，通常采用蒙特卡洛方法进行数值近似：

$$\int_{\Omega} f(\omega) d\omega \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(\omega_i)}{p(\omega_i)}$$

为了提高收敛速度，通常按照BRDF值的分布进行重要性采样，即生成的采样点$\omega_i$的概率密度函数$p(\omega_i)$与BRDF成正比。

## 6. 总结与应用前景

BRDF和辐射度量学为计算机图形学提供了描述光与物质相互作用的严格数学框架，是现代真实感渲染的理论基础。随着计算机硬件的不断进步，基于物理的渲染已从离线渲染领域扩展到实时渲染，在游戏、VR/AR、产品可视化等领域获得广泛应用。

未来的发展方向包括：
- 更高效的重要性采样技术
- 复杂材质的多层BRDF模型
- 结合机器学习的BRDF压缩与重建
- 考虑次表面散射和波动光学效应的扩展模型

通过对BRDF的深入理解和应用，我们能够创造出更加真实、更具视觉冲击力的计算机生成图像，推动计算机图形学技术不断向前发展。 