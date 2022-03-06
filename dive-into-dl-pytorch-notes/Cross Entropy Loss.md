# Cross Entropy Loss

## 推导

交叉熵是信息论中的一个重要概念，它的大小表示两个概率分布之间的差异，可以通过最小化交叉熵来得到目标概率分布的近似分布。

![image-20220306115638753](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220306115638753.png)

其中p是真实值，q是预测值

其中p表示真实值，在这个公式中是one-hot形式；q是预测值，在这里假设已经是经过softmax后的结果了。

仔细观察可以知道，因为pp的元素不是0就是1，而且又是乘法，所以很自然地我们如果知道1所对应的index，那么就不用做其他无意义的运算了。所以在pytorch代码中target不是以one-hot形式表示的，而是直接用scalar表示。所以交叉熵的公式(**m表示真实类别**)可变形为：

![image-20220306115738768](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220306115738768.png)

**其实就是先求预测q的softmax再取log，其中第m个就是cross entropy loss(因为除了第m项其他都是0)**

即相当于F.cross_entropy 自动调用log_softmax和nll_loss来计算交叉熵，计算方式如下：

![image-20220306120039958](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220306120039958.png)

其中nll_loss全称为negative log likelihood loss，表达式为：

![image-20220306120231546](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220306120231546.png)

## pytorch函数

```python
class torch.nn.CrossEntropyLoss(input， target， weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
```

该函数计算了input和target之间的交叉熵

**input** tensor：(minibatch, C, d1, d2, ... , dk) ->images时k=2

**target** tensor：(minibatch, d1, d2, ... , dk) ->images时k=2

> target 会进行one-hot编码（编码后其数值不是0就是1），变成与input形状相同
>
> 自动调用softmax将input的数值归一化



可选参数**weight**应当是一个一维的张量，表示每个类别的权重，这在你训练数据集样本不均衡时十分有用

输入张量input应当是每个类对应的原生的、**未经过归一化**的概率

如果指定了**ignore_index**，那么即使这个索引不在class range的范围内，loss也会接收这个索引的类

**注意：当在target中指定类索引时性能会更好；只有当每个minibatch对应一个类标签的限制情况下才考虑将target设置成类概率**

size_average(已弃用)