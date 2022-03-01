# U2Net

## Abstract

two-level nested U-structure

得益于在我们提出的RSU（残差块）中混合着的不同尺寸的感受野，我们能够更好地捕获不同尺寸的上下文信息

在RSU中使用了池化操作，使得模型计算开销不会随着模型深度加深而剧烈变化

这种模型架构是我们无需使用图像分类任务的backbone

## Introduction

大部分现有的SOD网络有一个共同的模式：focus on 利用现有backbone（AlexNet，VGG，ResNet）提取的深度特征，然而这些backbone起初都是为图像分类设计的，它们提取代表语义的特征，而不是局部细节和全局对比度信息，但这里所缺失的对SOD却至关重要。

早期的网络为了提取多级特征，通常会采用更深的网络，同时为了减少计算量，又会降低图像的分辨率，然而高分辨率对图像分割实际上至关重要。

作者设计了U2Net（**两级嵌套**的U-structure）解决了上述两个问题：

> 在底层（U-structure），采用新型的残差块（RSU），能够提取多尺度特征，而不降低图像分辨率
>
> 在顶层（U-structure），采用每个stage都使用RSU块的U-Net结构

## Related Works

传统方法手工特征：基于前景一致性、高光谱信息、超像素相似性、直方图

多级深度特征聚合

## Proposed Method

### 1、Residual U-blocks

局部信息和全局信息对SOD任务十分重要，1x1和3x3的卷积核由于占内存少，计算量小，被广泛使用；但是小尺寸的卷积核无法捕获全局信息，浅层的特征图输出只包含局部特征，为了在浅层的高分图像上获得更多的全局特征，最简单的想法就是扩大感受野

Inception block通过空洞卷积扩大感受野，但计算量非常大

PoolNet提出金字塔池化模块（Pyramid pooling module，PPM）到那时不同尺度的特征在上采样时如果直接concat或者add会导致特征退化

我们提出了RSU模块，如下图RSU-L（Cin，M，Cout）

![image-20220301105400574](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220301105400574.png)

其中L表示encoder中的层数，Cin表示输入通道数，Cout表示输出，M表示RSU输入 \ 输出通道数

RSU可以主要被分为三个部分：

> :one:一个输入卷积层，实现局部特征提取，输出F1(x)
>
> :two:一个U-Net结构，将F1(x)作为输入，输出U(F1(x))
>
> 更大的L意味着更深的网络，更多的池化操作，更大的感受野，以及更丰富的局部、全局信息。在逐步的的下采样、先进的上采样、concatenate和conv提取多尺度特征，避免了直接大范围上采样带来的细节损失
>
> :three:残差连接：F1(x) + U(F1(x))

RSU和普通残差块的对比：

![image-20220301111215522](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220301111215522.png)

普通残差块：`H(x) = F2(F1(x)) + x`

RSU: `Hrsu(x) = U(F1(x)) + F1(x)`

主要区别在于RSU使用一个U-Net块代替了之前简单的卷积，**这里的U就是上面的RSU-L块**

这种设计使网络能够从多个尺度直接从每个residual block中提取特征

由于是U型结构，大多数操作都是下采样，计算开销很小，RSU与其他网络开销对比如下图（计算量随内部通道数M的变化）

PLN: Plain convolution block

RES: Residual Block

DSE: Dense block

INC: Inception block

RSU: Residual U-block

![image-20220301112602241](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220301112602241.png)

### 2、Architecture of U2-Net

堆叠多个U型结构的网络已经有人探索过，通常都是按照顺序叠加来简历级联模型 ，可称为`U x n-Net`,n代表重复U型模块的个数，但存在的问题是，其计算和内存也被放大了n倍

我们提出Un-Net, 理论上，指数n可以设置为任意正整数，以实现单级或多级嵌套U型结构。但是嵌套级别太多的体系结构将太复杂，无法在实际应用中实现和使用。 

![image-20220301114515708](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220301114515708.png)

top-level：外部大的U型结构，包括了11个stages（stage是上图中的立方体）

button-level：stage中的U型结构，每个stage中都包含一个RSU块

以上两个，top-level和button-level一起被称为“nested U-structure”——嵌套U型结构；可以更有效地的提取、聚合intra-stage multi-scale features

如上图所示，一个U2Net主要包括三个部分：

> :one:一个包含6个stages的encoder
>
> :two:一个包含5个stages的decoder
>
> :three:一个用于融合decoder stages和last encoder stage的显著图融合模块

随着分辨率下降，RSU的深度L逐渐变小：

> **encoder**
>
> En1->RSU-7
>
> En2->RSU-6
>
> En3->RSU-5
>
> En4->RSU-4
>
> En5->RSU-4F    F表示用空洞卷积代替池化层和上采样操作，如下图所示，也就是说RSU-4F中的所有feature maps尺寸均相同（分辨率已经很低了，再池化会使特征退化）![image-20220301151416346](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220301151416346.png)

> **decoder**
>
> decoder具有与encoder对称的相似性，同时还有采用跳跃连接

> **fuse**
>
> 最后是显著图融合模块，类似HED，U2Net首先生成6个side output显著图：Sside6，Sside5，Sside4，Sside3，Sside2，Sside1；它们分别来自En6、De5、De4、De3、De2、De1的输出再加上一个3x3的conv和sigmoid激活得到的结果。
>
> 然后将logits（conv后sigmoid前的图像）上采样到输入图像大小，再经过1x1 conv和sigmoid后将他们concatenate得到最终的显著性概率图

### 3、Supervision

**deep supervision [HED] [DSS]**

**其实就是网络的中间部分新添加了额外的loss，跟多任务是有区别的，多任务有不同的GT计算不同的loss，而深度监督的GT都是同一个GT，不同位置的loss按系数求和。**

**深度监督的目的是为了浅层能够得到更加充分的训练，避免梯度消失**

对每个输出的显著图都计算loss，并且具有不同的权重w；

![image-20220301154302387](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220301154302387.png)

其中loss为standard binary cross-entropy

![image-20220301154321544](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220301154321544.png)

## Experimental Results

### 1、datasets

training dataset: DUTS-TR(10553), 水平翻转扩充数据集（21106）

Evaluation datasets: DUT-OMRON(5168, 含1-2个复杂前景物体), DUTS-TE(5019), HKU-IS(4447, 多个前景目标), ECSSD(1000, 包含较大的前景目标), PASCAL-S(850, 背景杂乱), SOD(300, 难), 

### 2、Evaluation Metrics

深度显著目标检测方法的输出常常是与输入图像有相同分辨率的预测图，每个像素点要么是0要么是1（0\255）；

采用了6种评估方法：

> Precision-Recall (PR) curves
>
> maximal F-measure
>
> Mean Absolute Error(MAE)
>
> weighted F-measure
>
> structure measure
>
> relaxed F-measure of boundary

### 3、Implementation Details

train中将输入image先resize到320x320，再随机垂直翻转resize到288x288

所有卷积层由**Xavier**初始化

loss的权重均初始化设为1

使用Adam optimizer，采用默认参数lr=1e-3,betas=(0.9, 0.999), eps=1e-8, weight decay=0)

测试输出由320x320 resize到原图大小，这里采用双线性插值

训练时长：1080ti 120hours

### 4、消融实验

（感觉没啥好看的 都是实验比较）

## Conclusions

