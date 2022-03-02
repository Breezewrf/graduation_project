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

采用Encoder-Decoder结构，因为这样的架构既能捕获高层全局语义，也能捕获底层细节，为了减少过拟合，最后一层