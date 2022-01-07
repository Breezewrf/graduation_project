## 创建Tensor

`x = torch.empty(5, 2)`

`x = torch.rand(5, 2)`(不能加dtype参数)

`x = torch.zeros(5, 2)`

```python
通过现有tensor创建：
x = x.new_ones(5, 3, dtype=torch.float64)# 返回全1的tensor，默认与输入tensor有相同dtype和device
x = torch.randn_like(x, dtype=torch.float)# 可以指定新数据类型

y = torch.ones_like(x, dtype = torch.double,device = 'cuda')
```

`使用shape或者size() 可以获取Tensor的形状`

以下常见创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu)。

| 函数                              | 功能                        |
| --------------------------------- | --------------------------- |
| Tensor(*sizes)                    | 基础构造函数(默认全0初始化) |
| tensor(data,)                     | 类似np.array的构造函数      |
| ones(*sizes)                      | 全1Tensor                   |
| zeros(*sizes)                     | 全0Tensor                   |
| eye(*sizes)                       | 对角线为1，其他为0          |
| arange(s,e,step)                  | 从s到e，步长为step          |
| linspace(s,e,steps)               | 从s到e，均匀切分成steps份   |
| rand/randn(*sizes)                | 均匀/标准分布               |
| normal(mean,std)/uniform(from,to) | 正态分布/均匀分布           |
| randperm(m)                       | 随机排列                    |

## Tensor操作

加法形式

`x + y`

```python
torch.add(x, y)
or
result = torch.empty(5, 3)
torch.add(x, y, out=result)
```



`y.add_(x) #inplace操作 ;PyTorch操作inplace版本都有后缀, 例如x.copy_(y), x.t_()`

注：

> inplace 参数在很多函数中都会有，它的作用是：**是否在原对象基础上进行修改**
>
> ​	inplace = True：不创建新的对象，直接对原始对象进行修改，**返回为none**；
>
> ​	inplace = False：对数据进行修改，创建并**返回新的对象承载其修改结果**。
>
> 默认是False，即创建新的对象进行修改，原对象不变，和深复制和浅复制有些类似。

## 索引

注意：索引出来的结果与原数据**共享内存**，也即修改一个，另一个会跟着修改

y = x[0, :]

y += 1

x也将相应改变

## 改变形状

1、使用view()

**注意`view()`返回的新`Tensor`与源`Tensor`虽然可能有不同的`size`，但是是共享`data`的，也即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)**

```python
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
# torch.Size([5, 3]) torch.Size([15]) torch.Size([3, 5])
```

2、使用reshape()

**返回一个真正新的副本（即不共享data内存）；但是此函数并不能保证返回的是其拷贝，所以不推荐使用。推荐先用`clone`创造一个副本然后再使用`view`**

**使用`clone`还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源`Tensor`。**

>  **另外一个常用的函数就是`item()`, 它可以将一个标量`Tensor`转换成一个Python number**

## 广播机制

当对两个形状不同的`Tensor`按元素运算时，可能会触发广播（broadcasting）机制：先适当**复制**元素使这两个`Tensor`形状相同后再按元素运算。

## 运算的内存开销

索引操作是不会开辟新内存的，而像`y = x + y`这样的运算是会新开内存的，然后将`y`指向新内存。

使x + y 的运算结果继续写入y的方法：

１、如果想指定结果到原来的`y`的内存，我们可以使用前面介绍的索引来进行替换操作。可以将`x + y`的结果通过`[:]`写进`y`对应的内存中。

​		`y[:] = y + x`

2、使用运算符全名函数中的`out`参数

​		`torch.add(x, y, out = y)`

3、自加运算符

​		`y += x`

## Tensor与Numpy相互转换

我们很容易用`numpy()`和`from_numpy()`将`Tensor`和NumPy中的数组相互转换。但是需要注意的一点是： **这两个函数所产生的的`Tensor`和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！**

Tensor -> Numpy

```python
a = torch.ones(5)
b = a.numpy()
```

Numpy -> Tensor

```
a = np.ones(5)
b = torch.from_numpy(a)
```

另外可以使用**torch.tensor()**将numpy转换为tensor,但注意该方法会进行数据拷贝,返回的tensor与原来的数据不再共享内存,速度会慢很多.

`c = torch.tensor(a)`

## Tensor on GPU

相关函数:

​	`torch.cuda.is_available`

​	`device = torch.device("cuda")`

​	`y = torch.ones_like(x, device=device)`

​	`x = x.to(device) #等价于x = x.to("cuda")`