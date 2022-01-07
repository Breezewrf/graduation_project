## 自动求梯度

将tensor 的属性`.requires_grad`设置为`True`时(创建tensor时需要显性设置,或者通过`.requires_grad_()`来用in-place的方式改变`requires_grad`属性 即`a.requires_grad_(True)`),它将开始追踪(track)在其上的所有操作

这样就可以利用链式法则进行梯度传播了。完成计算后，可以调用`.backward()`来完成所有梯度计算。此`Tensor`的梯度将累积到`.grad`属性中。

> 注意在`y.backward()`时，如果`y`是标量(scalar,无方向)，则不需要为`backward()`传入任何参数；否则，需要传入一个与`y`同形的`Tensor`

> 如果不想要被继续追踪，可以调用`.detach()`将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。

> 还可以用`with torch.no_grad()`将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（`requires_grad=True`）的梯度。

> `Function`是另外一个很重要的类。`Tensor`和`Function`互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。
>
> 每个`Tensor`都有一个`.grad_fn`属性，该属性即创建该`Tensor`的`Function`, 就是说该`Tensor`是不是通过某些运算得到的，若是，则`grad_fn`返回一个与这些运算相关的对象，否则是None。

:star:**不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量**。所以必要时我们要把张量通过将所有张量的元素加权求和的方式转换为标量，举个例子，假设`y`由自变量`x`计算而来，`w`是和`y`同形的张量，则`y.backward(w)`的含义是：先计算`l = torch.sum(y * w)`，则`l`是个标量，然后求`l`对自变量`x`的导数。

![img1](./images/Note2_img1.jpg)