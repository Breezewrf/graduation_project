# UNet

不需要数千个大量的训练样本，UNet依赖强大的数据增强功能，该体系结构包括一个用于捕获上下文的收缩路径和一个能够精确定位的对称扩展路径。这种网络可以从很少的图像中进行端到端训练，速度很快。

卷积网络的典型用途是分类任务，其中图像的输出是单个类别标签。然而，在许多视觉任务中，尤其是在生物医学图像处理中，期望的输出应该包括定位，即，应该为每个像素分配一个类别标签。此外，在生物医学任务中，通常很难拿到数千张训练图像。

**滑动窗口**方法：在滑动窗口中训练一个网络，通过提供像素周围的局部区域（补丁）作为输入来预测每个像素的类别标签；优点：网络能够定位，且在patches部分的训练数据远大于训练图像个数（相当于扩增了数据集）。缺点：由于要分别关注每个patch，网络速度非常慢并且在重叠的patch上会存在冗余；在定位准确性和语境（context）使用之间存在权衡：更大的patch需要更多的最大池化层，这使得定位精确度降低，然而小的patch允许网络只看到很少的context

## Network Architecture

![image-20220302101143546](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220302101143546.png)

contracting path

> 2个3x3conv  RELU  2x2maxpooling  double channels

expansive path

> up-conv halve channels concatenation 2个3x3conv RELU

使用加权损失，背景标签会获得较大的权重

边界采用镜像填充

采用弹性形变来进行数据增强，因为在实际的医疗情景中变形的目标是很常见的

## Training

为了最小化开销并最大限度地利用GPU内存，我们更喜欢大的输入块而不是大的批处理大小，从而将**批处理减少到单个图像**

使用SGD，**高动量(0.99)**，这样，大量之前看到的训练样本决定了当前优化步骤中的更新。

### 能量函数（Loss）

cross entropy：

![image-20220302112514305](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220302112514305.png)

:star:其中w是权重，我们将预先计算每个ground truth segmentation 的权值图，以补偿训练数据集中某类像素的不同频率, weighted map可按如下公式计算

![image-20220302113004991](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220302113004991.png)

d1：the distance to the border of the nearest cell

d2：the distance to the border of the second nearest cell

set w0 = 10, sigma = 5

### 权重初始化

:star:对于UNet的网络（交替卷积和ReLU），可以通过从标准偏差为`sqr(2/N)`的高斯分布中绘制初始权值来实现，其中N表示一个神经元的传入节点数。例如，对于前一层的3x3卷积和64个特征通道N = 9·64 =  576。

注：此方法来自[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.pdf (arxiv.org)](https://arxiv.org/pdf/1502.01852.pdf)