# BASNET

## 1、Introduction

**Boundary-Aware Salient Object Detection**

更加关注边界质量

全新的hybrid loss

prediction + refinement

densely supervised Encoder-Decoder **prediction**+ residual **refinement**

 hybrid loss 融合了二值交叉熵（BCE）、结构相似度（SSIM）和交并比（IoU）的loss，在hybrid loss的引导下能够有效地分割出显著目标区域，准确地预测出边界清晰的精细结构。

在SOD中，全局信息很重要，大多数网络只选用交叉熵（CE）作为loss，但CE在区分边界像素时可信度较低；其他的loss比如IoU、F-measure、Dice-score，但他们并不是为了捕捉细节设计的

## 2、Related Works

## 3、BASNET

采用Encoder-Decoder结构，因为这样的架构既能捕获高层全局语义，也能捕获底层细节，为了减少过拟合，每个decoder的最后一层都采用HED的deep supervise

定义了一下**coarse**的概念：

![image-20220303212905171](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220303212905171.png)

横轴表示每个像素点，纵轴表示该像素点为“真”的概率，当所有像素点只能概率只能取0/1时代表ground truth(图a)，图b表示的是coarse边界与gt不够拟合的情况，图c表示的是coarse区域的probability太低了，图d表示的是同时具备b、c的情况，也就是我们所指的“**coarse**”