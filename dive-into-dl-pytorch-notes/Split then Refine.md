# Split then Refine

基于BVMR的ResUNet，使用channel attention共享3个decoder的所有参数所学的bias

attention_guided

感知损失函数deep perceptual loss、SSIM loss

two_stage 

​	先SplitNet采用多任务学习，**domain(task)-specific attention**（对每个不同的task分别使用multi channel attention）；

​	再用RefineNet处理predicted mask和coarser restored image使用mask-guided spatial atttention（spatial-separated attention module）

## SplitNet

<font color="red">Q: consider the joint learning framework as a multi-domain learning problem</font>

<font color="blue">A: 训练数据来源于多个domain，domain information被纳入学习过程。是自然语言处理领域一个常见的学习方法（例如用在多个产品的情感分类和多个用户的垃圾邮件过滤等课题中），但很少有人应在计算机视觉领域。</font>

**首次将joint learning 任务看作multi-domain learning**

在**multi-domain learning**中，为了学习一个有效的模型，几乎所有的参数都在训练过程中共享，而每个领域需要使用不同的参数来强调。同样，我们框架中的三个任务侧重于学习一个空间区域，每个任务都必须学习其特定的特征，以便进行个体重建。具体来说，我们分析所有decoder中高三层的可学习参数，并采用**domain attention**分别学习每个任务的特定特征

受到**SE-Net**的启发，我们设计task-specific attention来re-weight每个decoder(task)中channels的重要性（即multi channel attention）

思路：先用ResBlocks学习三个任务的基本特征，然后再用task-specific attention来re-weight,具体如下图：

![image-20220228211332660](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220228211332660.png)