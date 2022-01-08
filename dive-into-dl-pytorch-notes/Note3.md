## 线性回归

### 手动实现

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)
        
def linreg(X, w, b):  
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):  
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):  
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
  
# 初始化数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                          dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float32)
# 初始化权重
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 设置requires_grad
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 

# 设置超参数
lr = 0.03
num_epochs = 3

# 开始训练
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        
        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
        
    train_l = squared_loss(linreg(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
 
# 训练完成
print(true_w, '\n', w)
print(true_b, '\n', b)
```

补充:

```
torch.mul(a, b) 是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是	(1, 2)，返回的仍是(1, 2)的矩阵；
torch.mm(a, b) 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 	3)的矩阵。
torch.bmm() 强制规定维度和大小相同
torch.matmul() 没有强制规定维度和大小，可以用利用广播机制进行不同维度的相乘操作
```

```
torch.index_select(input,dim,index) ==== input.index_select(dim,index)

torch.index_select(input, dim, index, out=None) 函数返回的是沿着输入张量的指定维度的	指定索引号进行索引的张量子集，其中输入张量、指定维度和指定索引号就是 torch.index_select(input, dim, index, out=None) 函数的三个关键参数，函数参数有：

input(Tensor) - 需要进行索引操作的输入张量；
dim(int) - 需要对输入张量进行索引的维度；
index(LongTensor) - 包含索引号的 1D 张量；
out(Tensor, optional) - 指定输出的张量。比如执行 torch.zeros([2, 2], out = tensor_a)，相当于执行 tensor_a = torch.zeros([2, 2])；
```

### Pytorch实现

首先，导入`torch.nn`模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。

之前我们已经用过了`autograd`，而`nn`就是利用`autograd`来定义模型。`nn`的核心数据结构是`Module`，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。

**方法一:  **在实际使用中，最常见的做法是**继承`nn.Module`，撰写自己的网络/层**。一个`nn.Module`实例应该**包含一些层以及返回输出的前向传播（forward）方法**。

```python
# 方法一 继承nn.module
import torch
from torch import nn
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
```

**方法二:**  `Sequential`是一个有序的容器，网络层将按照在传入`Sequential`的顺序依次被添加到计算图中。

```python
# 方法二 使用nn.Sequential
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])

```

代码 :

```python
import torch
import numpy as np
from torch import nn
from torch.nn import init
import torch.optim as optim
import torch.utils.data as Data

# 初始化数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# data_iter
batch_size = 10
    # 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
    # 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 初始化神经网络
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 初始化权重
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

# 初始化loss
loss = nn.MSELoss()

#初始化优化器
optimizer = optim.SGD(net.parameters(), lr=0.03)

#开始训练
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

```

注意：`torch.nn`仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用`input.unsqueeze(0)`来添加一维。

我们通过调用`optim`实例的`step`函数来迭代模型参数。按照小批量随机梯度下降的定义，我们在`step`函数中指明批量大小，从而对批量中样本梯度求平均。

`net[0]`这样根据下标访问子模块的写法只有当`net`是个`ModuleList`或者`Sequential`实例时才可以

如何调整学习率呢？主要有两种做法。一种是修改`optimizer.param_groups`中对应的学习率，另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。