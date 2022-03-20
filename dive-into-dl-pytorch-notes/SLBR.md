# SLBR

## Intro

two-stage：粗糙阶段、细化阶段

**第一阶段**有两个分支：水印分支、背景分支，水印分支对粗略估计的掩模进行自校准，并将校准后的掩模传递给背景分支重建水印区域

**第二阶段**：采用融合多层次特征来提高水印区域的纹理质量

SLBR: 自校准定位和背景细化

![image-20220316155703322](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220316155703322.png)

![image-20220316155634476](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220316155634476.png)

## Module

**自校准掩膜细化**（SMR）

> mask decoder分支，得到预测掩码输出，loss采用l1 距离，SMR模块采用BVMR的思想，连续三个residual block堆叠
>
> SMR中的**自纠正算法是一个亮点**（Liu_Improving_Convolutional_Networks_With_Self-Calibrated_Convolutions_CVPR_2020_paper）

**掩码引导背景增强**（MBE）

> image decoder分支，得到coarse image输出，loss采用bce loss

**Coarse Stage**

就是简单的编码、解码，与BVMR类似

**Refinement Stage**

> 将Coarse Stage的两个输出掩膜mask和coarse原图I_c做concatenate，作为输入
>
> 首先使用三个encoder来提取多级特征，为了充分利用coarse stage修复的内容信息，我们在coarse stage阶段的decoder和refinement stage阶段的decoder之间添加了跳跃连接（参考WDNet：尽管WDNet[在细化阶段也使用粗略阶段特征，但它们只是将粗略阶段的最后一个特征映射附加到细化阶段的输入。**不同的是，我们以对称的方式将粗糙阶段的每个背景解码器特征与其在细化阶段具有相同空间大小的相应编码器特征连接，从而在细化阶段产生增强的多级编码器特征**。）

**跨级特征融合**（CFF）

> 首先说明：**低层**编码器拥有较**大**的空间尺寸，高层编码器拥有较小的空间尺寸
>
> 为了更好地利用多级编码器的特性，我们提出了CFF模块。在每个CFF模块中，我们会将高层编码器特征上采样到其他不同尺寸的低层特征（**稀疏连接**：只将高级特征传播到其他级别的特征，作者也在另一篇paper尝试过密集连接，效果不及稀疏连接），然后将他们concatenate（按元素加），再加上堆叠的residual块（上图解释的很详细了）
>
> 一共堆叠三个CFF块，最后将所有level的编码器特征调整为目标图像大小，并concatenate，使用1x1 conv进行特征映射生成细化的无水印图像
>
> loss也采用L1距离

## Loss

在ImageNet上预训练的VGG16的基础上使用了**感知损失(perception loss)**：<font color='red'>不是很懂</font>

![image-20220316163147278](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220316163147278.png)

最后的损失函数：

![image-20220316163538005](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220316163538005.png)

即coarse阶段guess-image的L1 loss、refine阶段guess-image的L1 loss、感知损失、两个mask掩膜的bce（注意看SMR那边，有两个掩膜mask输出）

## Dataset

Large-scale Visible Watermark Dataset (LVW) ：主要是灰色的水印，模式、形状较为单一，背景来自PASCAL VOC2012

Colored Large-scale Watermark Dataset (CLWD) ：由本论文首次提出，彩色、多样化，更加真实、具有挑战性