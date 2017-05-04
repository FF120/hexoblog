---
title: scikit-learn Supervised learning
toc: true
categories:
  - 机器学习
tags:
  - scikit-learn
  - python
date: 2016-06-18 19:24:28
---
监督学习一般是指已知问题的线索（特征）和答案（标记），学习它们之间的内在联系，然后根据另外的在训练的时候没有见过的线索，推测出答案的过程。
而无监督学习是指根本不知道正确答案，让算法自动发现隐藏在数据内部的规律。scikit-learn提供了很多监督算法的实现。
<!-- more -->

# 支持向量机（SVM）

## 简介
scikit-learn支持稠密的(dense)和稀疏的（sparse）数据，但是当测试数据是稀疏的时候，训练数据必须也是稀疏的。为了达到最优的性能，建议稠密数据使用`numpy.ndarray`,稀疏数据使用`scipy.sparse.csr_matrix` and `dtype=float64`.

### 用途
- 分类（classfication）
- 回归 (regression)
- 离群点检测 (outliers detection)

### 优点
- 当特征维数很高时很有效(effective in high dimensional space)
- 当特征的维数远大于样本的数量的时候依然有效
- 使用的是训练点（training points）的子集进行决策函数（decision function）的计算，所以是内存高效的（memory efficient）
- 多种可供选择的核函数(kernel function)提高了算法的灵活性，核函数是可以根据自己的需要自定义的。

### 缺点
- 当特征的数量远大于样本的数量的时候，算法的性能会下降（poor performance）
- SVM不直接提供概率估计（probability estimates），而是使用一个五折交叉验证（five-fold cross-validation）,计算复杂性较高，一般不适合海量数据的处理。

## 使用方法

### 二分类
```python
#准备数据
X = [[0, 0], [1, 1]]
y = [0, 1]
#引入支持向量机
from sklearn import svm
'''
创建模型,这里有三种方法:
svm.SVC(); svm.NuSVC(); svm.LinearSVC()
'''
clf = svm.SVC()
'''
训练数据，这里X是[n_samples,n_features],y是[n_labels]
'''
clf = clf.fit(X_train, y)  
#使用训练好的模型预测
y_predicted = clf.predict(X_test)

#获得训练好的模型的一些参数
>>> # get support vectors
>>> clf.support_vectors_
array([[ 0.,  0.],
       [ 1.,  1.]])
>>> # get indices of support vectors
>>> clf.support_ 
array([0, 1]...)
>>> # get number of support vectors for each class
>>> clf.n_support_ 
array([1, 1]...)
#get the params of the svm
>>>clf.coef_
```
>上面是最简单的支持向量机的使用方式，下一步还需要了解可以设置的各个参数是什么意思，如何设置，如何交叉验证，如何选择和函数。

### 多分类
```python
from sklearn import svm
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
# "one-against-one"
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
'''
one-against-one 就是一对一，假设这四类的名称为a,b,c,d.
则需要训练区分(a,b)(a,c)(a,d)(b,c)(b,d)(c,d)的6种模型，所以
one-against-one这种策略在做多分类问题的时候会生成n*(n-1)/2个模型，每个模型区分其中的两个类。
'''
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
'''
 "one-vs-the-rest" 就是一对余下所有的，假设四类的名称为a,b,c,d;
 则需要训练区分(a,bcd),(b,acd)(c,abd)(d,abc)的4种模型，每个模型区分其中一个类，被除此类之外的所有类当作另外一个类处理。
 这种策略在做多分类问题的时候会生成n个模型。
'''

clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes

```
> 一些补充说明：`SVC`和`NuSVC`实现了`one-against-one`(`ovo`)方法，` LinearSVC`实现了`one-vs-test`(`ovr`)和另外一个叫做`Crammer and Singer`的实现多分类的方法，
可以通过指定`multi_class='crammer_singer'`来使用它。不多实践证明，在使用` LinearSVC`的进行多分类的时候，优先选择`one-vs-test`(`ovr`)，
因为`one-vs-test`(`ovr`)和`crammer_singer`得到的结果差不多，但是前者的计算时间要短。

### 模型参数说明

#### LinearSVC
```python 
from sklearn import svm
X = [[0,1],[2,3]]
y = [0, 1]
clf = svm.LinearSVC()
clf.fit(X,y)
print clf

>>>
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
```

**参数**
- `C`：可选参数，类型`float`,默认为1.0；误差的惩罚参数
- `class_weight`: 类型`dict`,可选参数，默认每个class的权重都是1.用来设置每个class的权重。
- `dual`:默认为`True`,类型`bool`,当`n_samples` > `n_features`时，设置成`False`.
- `fit_intercept`: 可选参数，类型为bool,默认为True. 意思是为模型计算截距（intercept），当数据事先已经是centered的时候，可以设置成False，不计算截距。
- `intercept_scaling`： 可选参数，类型为float,默认为1.意思是截距是否缩放。
- `loss`: 类型string,只能取"hinge" 和 "squared_hinge",默认取"squared_hinge"；定义SVM的损失函数，"hinge"是标准的SVM损失函数，"squared_hinge"是标准损失函数的平方。
- `max_iter`： 类型为int,默认为1000，模型最大的迭代次数。
- `multi_class`：类型string，只能取'ovr' 和 'crammer_singer' (默认值是'ovr')，当计算多分类的时候，指定多分类采取的策略。‘ovr’是将其中一类和剩下所有类二分，默认用这个策略就好。
- `penalty`： 类型string,只能取'l1' or 'l2' (默认值是'l2')，l1使参数稀疏，l2使大部分参数接近为0但是不是0，详细信息参考“机器学习中的范数”
- `random_state`： 只能取int seed, RandomState instance,  None 三个中的一个，默认值是None,指定产生伪随机数的时候使用的种子（seed）
- `tol`：可选参数，类型为float,默认值是1e-4,指定停止时候的允许的误差。
- `verbose`：类型为int,默认值是0，是否开启详细的输出，默认不要开启就好。如果开启，在多线程的时候可能运行不正确。

**属性**
- `coef_`：训练好之后的SVM模型中的参数的取值（就是系数），当是二分类的时候，shape=[n_features],多分类的时候，shape = [n_classes,n_features]
- `intercept_`:截距，二分类的时候shape = [1] ,多分类的时候shape=[n_classes]

#### SVC

```python
from sklearn import svm
X = [[0,1],[2,3]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X,y)
print clf

>>>
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
  
```

**参数**
- `C`=1.0, 可选参数，类型`float`,默认为1.0；误差的惩罚参数
- `cache_size`=200, 定义模型计算时使用的缓存大小，单位MB。
- `class_weight`=None,类型dict,默认为None,可以设置成'balanced'，这样会根据y自动计算每个class的权重。还可以手动设置每个class的权重。
- `coef0`=0.0,可选参数，类型为float,默认为0.0，独立于核函数（kernel function）的参数，只在'poly' and 'sigmoid'的时候有影响。
- `decision_function_shape`=None, 'ovo', 'ovr' or None, default=None
- `degree`=3, 可选参数，类型为int,默认为3，多项式和函数的度，其他类型的和函数自动忽略该参数。
- `gamma`='auto', 可选参数，类型为float,默认为‘auto’,默认取 1/n_features作为gamma的值。
- `kernel`='rbf',可选参数，类型为string，默认值为‘rbf’,定义SVM所使用的核函数，可选择的项如下：
	- linear
	- poly
	- rbf
	- sigmoid
	- precomputed
	- a callable(一个回调函数)
- `max_iter`=-1, 最大迭代次数，默认为-1，意思是无限制。
- `probability`=False, 可选参数，类型bool,默认值为False. 是否进行概率估计，使用之前需要先调用fit方法。
- `random_state`=None, 只能取int seed, RandomState instance,  None 三个中的一个，默认值是None,指定产生伪随机数的时候使用的种子（seed）
- `shrinking`=True,可选参数，类型boolean,默认值为True,是否开启“shrinking heuristic”
- `tol`=0.001, 可选参数，类型为float,默认值是1e-4,指定停止时候的允许的误差。
- `verbose`=False，类型为int,默认值是0，是否开启详细的输出，默认不要开启就好。如果开启，在多线程的时候可能运行不正确。

**属性**
- `support_` : array-like, shape = [n_SV]，支持向量的下标
- `n_support_` : array-like, dtype=int32, shape = [n_class] 每个类的支持向量的个数。
- `support_vectors_` ：shape = [n_SV, n_features]，支持向量(SVM确定了一个分类超平面，支持向量就是平移这个超平面，最先与数据集的交点。)
- `dual_coef_` : array, shape = [n_class-1, n_SV] 在决策函数（decision function）中支持向量的系数
- `coef_` : array, shape = [n_class-1, n_features]，特征的权重，只在线性核的时候可用。
- `intercept_` : array, shape = [n_class * (n_class-1) / 2]，决策函数（decision function）中的常量。

#### NuSVC

```python
from sklearn import svm
X = [[0,1],[2,3]]
y = [0, 1]
clf = svm.NuSVC()
clf.fit(X,y)
print clf

>>>
NuSVC(cache_size=200, class_weight=None, coef0=0.0,
   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
   max_iter=-1, nu=0.5, probability=False, random_state=None,
   shrinking=True, tol=0.001, verbose=False)
   
```
**参数**
大部分都与`SVC`一样，只是使用了一个额外的参数控制支持向量（support vector）的个数。
- `nu`:可选参数，类型float，默认值是0.5，值必须要(0,1]之间。

**属性**
- `support_` : array-like, shape = [n_SV]，支持向量的下标
- `n_support_` : array-like, dtype=int32, shape = [n_class] 每个类的支持向量的个数。
- `support_vectors_` ：shape = [n_SV, n_features]，支持向量
- `dual_coef_` : array, shape = [n_class-1, n_SV] 在决策函数（decision function）中支持向量的系数
- `coef_` : array, shape = [n_class-1, n_features]，特征的权重，只在线性核的时候可用。
- `intercept_` : array, shape = [n_class * (n_class-1) / 2]，决策函数（decision function）中的常量。

### 查看训练好的模型的参数

## 决策函数（decision function）

## 核函数（kernel function）

优先使用‘rbf’调节参数，当特征的数量远远大于样本的数量的时候，考虑使用线性核函数。

---------------------------------
# 随机梯度下降（Stochastic Gradient Descen）
分类，回归
## 简介
随机梯度下降法适用于特征数据大于10的5次方，样本数量大于10的5次方的大规模数据的处理领域。

### 用途
可以处理大规模数据和稀疏数据。

### 优点
- 高效
- 易于实现

### 缺点
- 需要很多超参数
- 对特征的缩放敏感

## 使用方法
```python
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier()
clf.fit(X, y) #训练
clf.predict([[2., 2.]])  #预测

print clf
>>>
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)

>>>clf.coef_  #模型系数
>>>Out[31]: array([[ 9.91080278,  9.91080278]])

>>>clf.intercept_    #截距
>>>array([-9.99002993])
```

**参数**
- `alpha`=0.0001, 
- `average`=False, 
- `class_weight`=None, epsilon=0.1,
- `eta0`=0.0, 
- `fit_intercept`=True, 
- `l1_ratio`=0.15,
- `learning_rate`='optimal', 
- `loss`='hinge', 
- `n_iter`=5, n_jobs=1,
- `penalty`='l2', 
- `power_t`=0.5, 
- `random_state`=None, 
- `shuffle`=True,
- `verbose`=0, 
- `warm_start`=False

**属性**
- coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,n_features);Weights assigned to the features.

- intercept_ : array, shape (1,) if n_classes == 2 else (n_classes,);Constants in decision function.

# 最近邻方法（Nearest Neighbors）
如果一个样本在特征空间中的k个最相 似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别.
## 简介
scikit-learn实现了监督的和非监督的最近邻方法，决定最近邻的算法有`ball_tree`,`kd_tree`,`brute`,可以通过指定模型参数`algorithm`的值来指定到底使用哪一个算法。
主要功能是实现***分类***和***回归***。

## 用法

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
```

# 决策树（Decision Trees）

# 朴素贝叶斯（Native Bayes）
