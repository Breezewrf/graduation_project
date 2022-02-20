## Blind Visual Motif Removal from a Single Image 论文阅读笔记

​		传统visual motif remove 的方法通常是分为两个步骤，先检测水印位置再进行去除，依赖于破坏像素位置信息，本文首次提出blind端到端方法进行motif水印去除，将corrupted image分解为background和visual motif的方法更有利于重建图像，这是受到了“Generative single image reflection separation ”的启发（去除反射）

数据集：由coco合成

网络结构：借鉴了UNet，有一个encoder，三个decoders

### Unet

### Residual

### Transpose Convolution

**上采样有3种常见的方法：双线性插值(bilinear)，反卷积(Transposed Convolution)，反池化(Unpooling)**

卷积尺寸计算公式：`o = (i + 2*p + s - k)/2`

反向卷积尺寸计算公式：

> ```python
> if (o + 2*p - k) % s == 0:  
> 	o = s*(i - 1) - 2*p + k
> else:
> 	o = s*(i - 1) - 2*p + k + (o + 2*p - k) % s
> ```

### hard example mining

inspiration -- [15]

[5] spatial perturbation

