链接导航栏

*课程github地址：https://github.com/datawhalechina/thorough-pytorch 
*课程gitee地址：https://gitee.com/datawhalechina/thorough-pytorch 
*B站视频：https://www.bilibili.com/video/BV1L44y1472Z

## 1.学习PyTorch环境搭建

这里分享几个安装torch相关库常用的链接，也方便日后再查看

https://download.pytorch.org/whl/torch/

https://download.pytorch.org/whl/torchvision/

https://www.lfd.uci.edu/~gohlke/pythonlibs/

![image-20220712193409596](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220712193409596.png)

还有切换cuda版本的指令

```sh
#<<< config root's cuda  start <<<
export PATH=/usr/local/cuda-11.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.0/bin:$CUDA_HOME
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# cuda-10.2
#<<< config root's cuda  end <<<
```

## 2.torch基础知识

要阻止一个张量被跟踪历史，可以调用`.detach()`方法将其与计算历史分离

`Tensor `和` Function` 互相连接生成了一个无环图 (acyclic graph)，它编码了完整的计算历史

注意grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。

## 3.数据加载方式总结

（1）使用torchvision的ImageFolder 方法

```python
import torch
from torchvision import datasets
train_data = datasets.ImageFolder(train_path, transform=data_transform)
val_data = datasets.ImageFolder(val_path, transform=data_transform)
```

（2）继承Dataset类+torch.utils.data的DataLoader方法

定义的类需要继承PyTorch自身的Dataset类。主要包含三个函数：

- `__init__`: 用于向类中传入外部参数，同时定义样本集
- `__getitem__`: 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
- `__len__`: 用于返回数据集的样本数

构建好Dataset后用DataLoader读入数据

```python
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
```

上述使用Dataloader加载的数据可以转化为迭代器，使用next来读取（ImageFolder加载的数据也可以）

```python
images, labels = next(iter(val_loader))
print(images.shape)
plt.imshow(images[0].transpose(1,2,0))
```