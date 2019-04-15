---
title: 深度学习_Theano使用技巧
toc: true

tags:
  - ML
  - theano
date: 2017-03-22 16:19:50
---

## theano 介绍
Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
(http://deeplearning.net/software/theano/)

<!-- more -->

## theano 安装
（http://deeplearning.net/software/theano/install_windows.html#install-windows）
Windows平台成功的安装方式，先安装anaconda, 然后执行
`$ conda install mingw libpython`

**错误信息**
. collect2.exe: error: ld returned 1 exit status

## theano 基本操作

### 引入常用的包
```python
from theano import *
import theano.tensor as T
## 计算卷积的函数
from theano.tensor.nnet import conv
```

### 函数的定义和计算
```python
# 标量
x = T.dscalar('a')
y = T.dscalar('b')
z = x * y ** 2
f = function([x,y],z)
# 向量
x = T.vector('a')
z = x + x
f = function([x],z)
f([1,2,3])
# 矩阵
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x,y],z)
f(np.array([[1,2,3],[4,5,6]]),np.array([[3,4,5],[6,7,8]]))

# 逻辑回归函数
# 验证 1 / (1 + T.exp(-x))   == (1 + T.tanh(x/2)) / 2
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
f = function([x],s)
f(np.array([[4,-1],[2,3]]))

s2 = (1 + T.tanh(x/2)) / 2
f2 = function([x],s2)
f2(np.array([[4,-1],[2,3]]))


## 一次计算多个函数
a,b = T.dmatrices('a','b')
f1 = a + b
f2 = a - b
f3 = a*b
f4 = a**b
f = function([a,b],[f1,f2,f3,f4])
ss = f(np.array([[1,2],[3,5]]),np.array([[1,2],[3,5]]))

# 为输入参数设置默认值
from theano import In
x,y = T.dscalars('x','y')
z = x + y
f = function([x,In(y,value=1)],z)
# 只传入x， y的默认值就是1
f(10)

## 截止到目前为止，变量都只是在函数内部使用，没有共享
# 下面看看如何实现值的累加
from theano import shared
count = shared(0) # 计数值初始化为0
inc = T.iscalar('inc')
f = function([inc],count,updates=[(count,count+inc)])
f(10)
f(2)
count.get_value()
count.set_value(-1)
count.get_value()

## 随机数
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2)) # 均匀分布的随机数
rv_n = srng.normal((2,2)) # 正太分布的随机数
f = function([],rv_u)
g = function([],rv_n,no_default_updates = True)
nearly_zeros = function([],rv_u + rv_u-2 * rv_u)

f()
g()
nearly_zeros()
# 设置随机数发生器的种子
rng_val = rv_u.rng.get_value(borrow=True)
rng_val.seed(100)
rv_u.rng.set_value(rng_val,borrow=True)
```

### 求函数的导数
```python
import numpy as np
import theano
import theano.tensor as T
from theano import pp
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y,x)
pp(gy)

f = theano.function([x], gy)
f(4)  # x的平方的导数
```

### 卷积操作

#### 2D卷积

theano计算卷积的函数：

```python
from theano.tensor.nnet import conv2d
output = conv2d(
    input, filters, input_shape=(1, 1, 5, 5), filter_shape=(1, 1, 3, 3),
    border_mode=(1, 1), subsample=(2, 2))
```

上式计算的是类似下面这样的卷积：

![](2017-05-12_161131.png)

其中各个参数的含义是：

- input(batch size, input channels, input rows, input columns)

>batch size : 一次处理的样本数量
input channels : input feature map 的数量
input rows: input feature map 的行
input columns: input feature map 的列

- filters(output channels, input channels, filter rows, filter columns).

>output channels ： output feature map 的数量
input channels : input feature map 的数量
filter rows ：卷积核的行
filter columns : 卷积核的列

- input_shape(batch size (b), input channels (c), input rows (i1), input columns (i2))

>batch size : 一次处理的样本数量
input channels : input feature map 的个数
input rows: input feature map 的行
input columns : input feature map 的列

- filter_shape(output channels (c1), input channels (c2), filter rows (k1), filter columns (k2))

>output channels ： output feature map 的个数
input channels : input feature map 的个数
filter rows & filter columns ： 卷积核的大小

- border_mode:	'valid', 'half', 'full' or (p_1, p_2)

>边缘补0的模式。

 - subsample:	(s1, s2)

 >定义卷积核的步长。
