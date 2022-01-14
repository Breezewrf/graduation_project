## 李宏毅《机器学习》task3

### 一、讨论了模型误差的主要来源有两个，一个是bias一个是variance

> 在简单的模型中，预测的结果比较集中，离散程度较小，故variance较小，但bias较大，发生欠拟合（Underfitting）；
>
> > Solution：此时应当重新设计模型，盲目增加数据集是不可取的。

> 模型过于复杂时，variance较大，bias较小，发生过拟合（Overfitting）
>
> > Solution1：more data（旋转、裁剪、拼接、镜像等）
> >
> > Solution2：regularization

### 二、cross validation

> N-fold Cross Validation

### 三、Gradient Descent tips

> 自适应学习率
>
> Adagrad

> 随机梯度下降（Stochastic Gradient Descent）

> 特征缩放（Scale）

### 四、梯度下降的泰勒证明