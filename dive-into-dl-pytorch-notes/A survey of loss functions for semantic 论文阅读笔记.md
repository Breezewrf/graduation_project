# A survey of loss functions for semantic论文阅读笔记

[TOC]

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329210228759.png" alt="image-20220329210228759" style="zoom:67%;" />

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329210255936.png" alt="image-20220329210255936"  />

## 摘要

图像分割已经成为一个活跃的研究领域，因为它具有广泛的应用，从自动疾病检测到自动驾驶汽车。在过去的五年里，不同的论文提出了不同的客观损失函数，用于不同的情况，如bias data，sparse segmentation(稀疏分割)等。在本文中，我们总结了一些众所周知的图像分割中广泛使用的损失函数，并列出了使用它们可以帮助模型更快更好收敛的情况。此外，我们还引入了一种新的损失函数：**log-cosh dice loss** ，并将其在NBFS头骨分割开源数据集上的性能与广泛使用的损失函数进行了比较。我们还展示了某些损失函数在所有数据集上都能很好地执行，并且可以在未知的数据分布场景中作为一个很好的baseline选择。

指标术语：计算机视觉，图像分割，医学图像，损失函数，优化，医疗保健，颅骨分离，深度学习

## 1、Introduction

深度学习已经彻底改变了从软件到制造业的各个行业。医学界也从深度学习中受益。在疾病分类方面已经有了多种创新，例如，使用U-Net的肿瘤分割和使用SegNet的癌症检测。图像分割是深度学习领域对医学领域的重要贡献之一。除了告诉人们某些疾病的存在，它还展示了它确切存在的地方。它极大地帮助创建了各种类型的医疗扫描中检测肿瘤、损伤等的算法。

图像分割可以定义为像素级的分类任务。图像由不同的像素组成，这些像素组合在一起定义了图像中的不同元素。将这些像素划分为元素的方法称为语义图像分割。在设计基于复杂图像分割的深度学习体系结构时，损失/目标函数的选择是非常重要的，因为它们触发了算法的学习过程。因此，从2012年开始，研究人员尝试使用各种领域特定的损失函数来改善他们的数据集的结果。在本文中，我们总结了 **15** 个这样的基于分割的损失函数，它们已经被证明在不同的领域提供了最新的结果。这些损失函数可以分为 **4类** :基于分布的、基于区域的、基于边界的和复合的(参见I)。我们还讨论了确定哪种目标/损失函数在场景中可能有用的条件。除此之外，我们还提出了一个新的log-cosh dice loss用于语义分割。为了展示其效率，我们在NBFS颅骨剥离数据集[1]上比较了所有损失函数的性能，并以Dice系数、灵敏度和特异性的形式分享了结果。代码实现可在GitHub:  https://github.com/shruti-jadon/SemanticSegmentation-Loss-Functions获得。

## 2、 损失函数

深度学习算法采用随机梯度下降法对目标进行优化和学习。为了准确和快速地学习目标，我们需要确保目标的数学表示，作为函数，它甚至可以覆盖边缘情况。损失函数的引入源于传统的机器学习，这些损失函数是在标签分布的基础上推导出来的。例如，binary cross entropy由伯努利分布推导而来，categorical cross entropy由多努利分布推导而来。在本文中，**我们关注的是语义分割而不是实例分割，因此像素级类的数量被限制在2个(实例分割要输出目标名称)**。在这里，我们将介绍15个广泛使用的丢失函数，并了解它们的用例场景。

### A. Binary Cross-Entropy

Cross Entropy[4]被定义为一个给定的随机变量或一组事件的两个概率分布之间的差异的度量。它被广泛用于分类目标，分割是像素级分类，效果很好。

Binary Cross-Entropy 被定义为：

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220324172703693.png" alt="image-20220324172703693" style="zoom: 67%;" />

### B. Weighted Binary Cross-Entropy（WBC）

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220324173202579.png" alt="image-20220324173202579" style="zoom:67%;" />

通过设置权重来均衡正负样本比例

### C. Balanced Cross-Entropy (BCE)

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220324173407124.png" alt="image-20220324173407124" style="zoom:67%;" />

与WBC略有差别，这里β是有具体定义的：

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220324173605775.png" alt="image-20220324173605775" style="zoom:50%;" />

### D. Focal Loss

是Binary Cross Entropy的变种，由于Binary Cross Entropy中y不是0就是1，可以简化为

`Lbce = -log(p)`

而FL就是在其基础上加权使其能够处理困难样本（解决正负样本不均衡问题）

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329195314743.png" alt="image-20220329195314743" style="zoom: 67%;" />

### E. Dice Loss

Dice是医学图像比赛中使用频率最高的度量指标，它是一种集合相似度度量指标，通常用于计算两个样本的相似度，值域为[0, 1]。在医学图像中经常用于图像分割，分割的最好结果是1，最差时候结果为0.

（好像下面这个公式有问题）

![[公式]](./images/equation-1648559425504.svg)

应该是这个公式：

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5C20200318134203976.png" alt="在这里插入图片描述" style="zoom:67%;" />

**与IoU有所不同，分母的重叠部分没有减掉而是直接在分子乘2。**

是计算机视觉界广泛使用的计算两幅图像之间相似度的方法

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329195418856.png" alt="image-20220329195418856" style="zoom:67%;" />

在这里，分子和分母都加了1，以确保函数在边缘情况下不是undefined(如y  =ˆp = 0)。

### F. Tversky Loss

可以看作是Dice系数的推广。它借助β系数对FP(假阳性)和FN(假阴性)增加了权重

### G. Focal Tversky Loss

类似于Focal  Loss，它通过弱化简单/常见的例子来关注困难的例子。focaltversky  loss[12]也尝试学习一些困难的例子，例如在小roi(感兴趣区域)的帮助下

### H. Sensitivity Specificity Loss

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329201709837.png" alt="image-20220329201709837" style="zoom:67%;" />

### I. Shape-aware Loss

形状感知缺失，顾名思义是考虑到形状的。一般来说，所有的损失函数都是在像素级工作的，而Shape-aware loss计算的是预测分割曲线到ground  truth周围点之间的平均点到曲线的欧氏距离，并将其作为交叉熵损失函数的系数。

![image-20220329201936653](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329201936653.png)

### J. Combo Loss

定义为Dice loss和modified cross entropy的加权和。它试图利用Dice的针对类别不平衡的灵活性，同时使用交叉熵来平滑曲线。

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329202236465.png" alt="image-20220329202236465" style="zoom:67%;" />

### K. Exponential Logarithmic Loss

指数对数损失：重点关注那些不太准确的结构，利用Dice loss和Cross Entropy loss的组合来实现。对Dice loss和Cross Entropy loss进行指数和对数变换，以便结合更精细的决策边界和准确的数据分布的优点

### L. Distance map derived loss penalty term

距离地图(Distance map)可以定义为ground truth与predict map之间的距离(欧氏距离、绝对距离等)。有两种方法可以结合距离图，一种是创建神经网络结构，在其中有一个重建头部(reconstructed head)和分割，或者将其引入损失函数。按照同样的理论，Caliva等人使用了来自ground truth mask的距离图，并创建了一个基于惩罚的自定义损失函数。使用这种方法，**很容易将网络的焦点引导到难以分割的边界区域**。

<img src="C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220329203202390.png" alt="image-20220329203202390" style="zoom:67%;" />

其中φ是生成的距离图

注意：**式子中加入的常数1可以解决U-Net结构中梯度消失问题。**

### M. Hausdorff Distance Loss

Hausdorff Distance (HD)是一种用在分割方法中用来跟踪模型性能的度量

### N. Correlation Maximized Structural Similarity Loss

许多基于语义的分割丢失函数只关注像素级的分类错误，而忽略了像素级的**结构信息**。其他一些损失函数已经尝试使用结构先验来添加信息，如CRF、gan等。在这个损失函数中，zhao等人引入了结构相似损失(Structural  Similarity loss, SSL)，以实现地面真实图和预测图之间的高度正线性相关。该方法分为结构比较、交叉熵权系数确定和小批量损失定义三个步骤。

### O. Log-Cosh Dice Loss

