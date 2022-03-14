# psnr|ssim

## psnr:

峰值信噪比经常用作图像压缩等领域中信号重建质量的测量方法，它常简单地通过均方误差（MSE）进行定义

![image-20220314221019215](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220314221019215.png)

![image-20220314221038489](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220314221038489.png)

> MAXI 是表示图像点颜色的最大数值，如果每个采样点用 8 位表示，那么就是 255；255的平方必然大于分母均方差，因此保证PSNR大于0，当MSE越小时，PSNR越大。单位是dB，数值越大表示失真越小，因为数值越大代表MSE越小。MSE越小代表两张图片越接近，失真就越小。

​		PSNR是最普遍和使用最为广泛的一种图像客观评价指标，然而它是基于对应像素点间的误差，即 **基于误差敏感的图像质量评价**。由于并未考虑到人眼的视觉特性（人眼对空间频率较低的对比差异敏感度较高，人眼对亮度对比差异的敏感度较色度高，人眼对一个区域的感知结果会受到其周围邻近区域的影响等），因而经常出现评价结果与人的主观感觉不一致的情况。

## **ssim:**

SSIM的全称为structural similarity index，即为结构相似性，是一种衡量两幅图像相似度的指标。该指标首先由德州大学奥斯丁分校的图像和视频工程实验室(Laboratory for Image and Video Engineering)提出。

公式：（其中x，y分别表示两张图像）

l: 照明度、c: 对比度、s: 结构

![image-20220314222939892](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220314222939892.png)

**可简化为**：

![image-20220314222908871](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220314222908871.png)



在实际应用中，一般采用高斯函数计算图像的均值、方差以及协方差，而不是采用遍历像素点的方式，以换来更高的效率。可以利用滑动窗将图像分块，令分块总数为N，考虑到窗口形状对分块的影响，采用高斯加权计算每一窗口的均值、方差以及协方差，然后计算对应块的结构相似度SSIM，最后将平均值作为两图像的结构相似性度量，即平均结构相似性MSSIM