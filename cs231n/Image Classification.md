## Image Classification

### Motivation

图像分类：为输入图像指定一个类别集的标签，是计算机视觉任务的核心。

例子：对于计算机而言，图像由三维数组构成，计算机将由这三维数组中的数据预测出图像类别。

挑战：视觉点误差；尺寸不同；形变；光线条件；背景相似；同类异样

数据驱动方法：与传统排序算法等不同

图像分类流程：输入 -> 学习 -> 评价

### Nearest Neighbor Classifier

这种方法和CNN还没有联系，目前实践中已经很少使用了

CIFAR-10：包括60000张32x32的小图片，一共有10类，被划分为50000张训练集和10000张测试集。

假如使用NN方法，给定一个50000张的训练集，希望能够预测剩余的10000张图像。NN方法将会对一张输入图片与训练集中的所有训练图片进行计算，并选取距离最近的作为预测标签。然而这种方法计算出的十个最近邻图片实际上只有三个是与输入图像同类别的，

基于L1距离的NN方法：

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred


Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)

print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```

## K-Nearest Neighbor Classifier

并非如NN方法一样只选择距离最小的那个类作为标签，KNN方法是选取距离最小的K个类别进行预测。

## Validation sets for Hyper parameter tuning

注意在选取超参数时不能使用到测试集，而是将训练集进行划分，选出验证集来调参。

交叉验证：当训练数据量较少时，我们使用交叉验证进行参数调整。与之前的方法相比，我们将不再随机选择1000个数据作为验证集，而是每次迭代不同的验证集。例如在五折交叉验证中，我们将训练集划分为五份，其中四份用于训练，一份用于验证。

实际使用：实践中人们常常使用的是单独验证集划分，而非交叉验证，因为交叉验证会造成昂贵的计算量。当超参数数量非常多时，验证集的数量要更大；如果验证集数量很少，建议使用交叉验证方法。

优缺点：

> 易于实现以及理解，无需训练，然而在测试用要花更多时间。
>
> 在另一方面，NN方法的计算复杂度是一个研究方向，研究人员提出Approximate Nearest Neighbor （ANN）算法来加速最近邻的查找。（依赖于预处理、预检索，建立kdtree或是k-means）
>
> 对于低维数据具有较大优势，但在图像分类上效果很差；像素值相近无法代表感知、语义相似



## KNN实践注意

1、 数据预处理，对数据进行归一化，但本节中图像中的像素大部分很相似，没有很广泛的分布，因此没有采用归一化

2、 如果数据具有高维特征，可以考虑使用降维的方法例如PCA ([wiki ref](https://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](https://web.archive.org/web/20150503165118/http://www.bigdataexaminer.com:80/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/)), NCA ([wiki ref](https://en.wikipedia.org/wiki/Neighbourhood_components_analysis), [blog ref](https://kevinzakka.github.io/2020/02/10/nca/)), or even [Random Projections](https://scikit-learn.org/stable/modules/random_projection.html)

3、 随机划分数据集为训练集和验证集

4、选用L1、L2距离

5、如果你的KNN跑的时间太长了，可以考虑使用Approximate Nearest Neighbor库(e.g. [FLANN](https://github.com/mariusmuja/flann)) 来进行加速，但会损失一些精度。

6、有一个问题是，在测试时是否使用完整的训练集？**因为将fold划入训练集后可能会导致产生不同的最佳超参数。因此我们将使用fold以外的train data进行测试。**

