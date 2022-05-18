# 记一个大Bug

在训练模型时会在前面加上：

`model.train()`
在测试模型时在前面使用：

`model.eval()`
同时发现，如果不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout。

> BN: 训练时是正对每个min-batch的，但是在测试中往往是针对单张图片，即不存在min-batch的概念。由于网络训练完毕后参数都是固定的，因此每个批次的均值和方差都是不变的，因此直接结算所有batch的均值和方差。所有Batch Normalization的训练和测试时的操作不同

> Dropout: 在训练中，每个隐层的神经元先乘概率P，然后在进行激活，在测试中，所有的神经元先进行激活，然后每个隐层神经元的输出乘P。

在转onnx的时候总是遇到输出很离谱的情况，首先排查输入图像的预处理是否有问题：

```python
@staticmethod
    def trans(*images):
        transformed = []
        for image in images:
            transformed.append((image / 127.5 - 1).astype(np.float32))
        return transformed
@staticmethod
    def flip(*images):
        flipped = []
        for image in images:
            flipped.append(np.transpose(image, (2, 0, 1)))
        return flipped
def load_image(image_path, _device, include_tensor):
    numpy_image = None
    tensor_image = None
    if os.path.isfile(image_path):
        to_save = False
        row_image = Image.open(image_path)
        w, h = row_image.size
        if h != 512 or w != 512:
            row_image = row_image.resize((512, 512), Image.BICUBIC)
        numpy_image = np.array(row_image)
        if len(numpy_image.shape) != 3:
            numpy_image = np.repeat(np.expand_dims(numpy_image, 2), 3, axis=2)
        if numpy_image.shape[2] != 3:
            numpy_image = numpy_image[:, :, :3]
        if include_tensor:
            tensor_image = trans(flip(numpy_image)[0])[0]
            tensor_image = torch.unsqueeze(torch.from_numpy(tensor_image), 0).to(_device)
        numpy_image = np.expand_dims(numpy_image / 255, 0)
    return numpy_image, tensor_image
```

这次查了好久，硬是没找到问题，最后代码逐行比对才发现差了一个eval

![image-20220411114535801](C:%5CUsers%5CBreeze%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220411114535801.png)

中间为eval后的输出，右边的是gt