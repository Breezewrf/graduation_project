## SPP-Net

| 读论文          | pass 1 （finished） | pass 2 | pass 3 |
| --------------- | ------------------- | ------ | ------ |
| 1. all titles   | √                   | √      | √      |
| 2. abstract     | √                   | √      | √      |
| 3. introduction |                     | √      | √      |
| 4. method       | graph               |        | √      |
| 5. experience   | graph               |        | √      |
| 6. conclusion   | √                   | √      | √      |

![image-20220326161557786](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cimage-20220326161557786.png)

红色框是selective search 输出的可能包含物体的候选框（ROI）。一张图图片会有~2k个候选框，每一个都要单独输入CNN做卷积等操作很费时；RCNN对输入图像大小固定限制为224x224，这可能会降低识别的精度。（事实上，卷积层不需要固定的图像大小，可以生成任何大小的特征图。另一方面，完全连接的层需要根据其定义有固定的大小/长度输入。因此，固定大小的限制只来自于完全连接的层，它们存在于网络的更深的阶段。）

SPP-net提出：能否在feature map上提取ROI特征，这样就只需要在整幅图像上做一次卷积；提出新的池化策略——空间金字塔池化（spatial pyramid pooling）加在全连接层之前，可以生成固定长度的特征表示而不考虑图像的规模大小。

![img](C:%5CUsers%5CBreeze%5CDesktop%5Cgra_proj%5Cgraduation_project%5Cdive-into-dl-pytorch-notes%5Cimages%5Cv2-7d5d06d0553d6dab106588ec4654df15_720w.png)

虽然总体流程还是 Selective Search得到候选区域 -> CNN提取ROI特征 -> 类别判断 -> 位置精修，但是由于所有ROI的特征直接在feature map上提取，大大减少了卷积操作，提高了效率。

<font color='blue'>（Fast R-CNN中还有一个ROI pooling，其作用和spatial pyramid pooling相同，实现方式有所不同，注意区分）</font>

[(78条消息) ROI Pooling 与 SPP 理解_梦星魂24的博客-CSDN博客](https://blog.csdn.net/qq_35586657/article/details/97885290)