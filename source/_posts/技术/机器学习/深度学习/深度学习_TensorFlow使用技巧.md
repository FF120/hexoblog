---
title: 深度学习_TensorFlow使用技巧
toc: true

tags:
  - TensorFlow
  - ML
date: 2017-05-12 17:27:52
---
TensorFlow 基础使用。

<!-- more -->

## 定义变量和执行运算：

```python
import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

## 运行程序
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
```

## 卷积

### 一维卷积

![](2017-05-12_192455.png)

```python
import numpy as np
import tensorflow as tf
# 输入的 feature_map
input_feature_map = np.array([0,1,2,-1,1,-3,0]).astype(np.float32)
# 卷积核
filter_kernel = np.array([1,0,-1]).astype(np.float32)

graph1 = tf.Graph()
with graph1.as_default():
    #inputs = [batch, in_width, in_channels]
    #filter =  [filter_width, in_channels, out_channels]
    # reshape成需要的输入格式
    f=tf.constant( input_feature_map.reshape(1,-1,1) )
    g=tf.constant(  filter_kernel.reshape(-1,1,1)   )
    conv1=tf.nn.conv1d( f,g, stride=1 , padding="VALID",name="conv1")

with tf.Session(graph=graph1) as sess:
    sess.run(tf.global_variables_initializer())
    result = (sess.run(conv1))
    print(result[0])
    sess.close()

    # 输出 [[-2.]
    #  [ 2.]
    #  [ 1.]
    #  [ 2.]
    #  [ 1.]]
```

用上面的代码计算的结果和例子中步长为1的一维卷积结果是一致的。把代码中的`stride=1`改为`stride=2`就可以得到例子中的另外一个结果：

```
[[-2.]
 [ 1.]
 [ 1.]]
```

### 二维卷积

```python
import numpy as np
import tensorflow as tf
# 输入的 feature_map
input_feature_map = np.array([[3,3,2,1,0],
                              [0,0,1,3,1],
                              [3,1,2,2,3],
                              [2,0,0,2,2],
                              [2,0,0,0,1]]).astype(np.float32)

filter_kernel = np.array([[0,1,2],
                       [2,2,0],
                       [0,1,2]]).astype(np.float32)

graph1 = tf.Graph()
with graph1.as_default():
    #inputs = [batch, in_height, in_width, in_channels]
    #filter =  [filter_height, filter_width, in_channels, out_channels]
    # reshape成需要的输入格式
    f=tf.constant( input_feature_map.reshape(1,5,5,1) )
    g=tf.constant(  filter_kernel.reshape(3,3,1,1)   )
    conv2=tf.nn.conv2d( f,g, strides=[1,1,1,1] , padding="VALID",name="conv1")

with tf.Session(graph=graph1) as sess:
    sess.run(tf.global_variables_initializer())
    result = (sess.run(conv2))
    print(result)
    sess.close()
```

得到的卷积结果：

![](2017-05-12_194134.png)

这里附上使用theano计算的相同的二维卷积的代码和结果：

```python
from theano import *
import theano.tensor as T
from theano.tensor.nnet import conv2d
import numpy as np

inputs = T.tensor4('input')
filters = T.tensor4('filters')
input_feature_map = np.array([[3,3,2,1,0],
                              [0,0,1,3,1],
                              [3,1,2,2,3],
                              [2,0,0,2,2],
                              [2,0,0,0,1]]).reshape(1,1,5,5).astype(float)

filter_map = np.array([[0,1,2],
                       [2,2,0],
                       [0,1,2]]).reshape(1,1,3,3).astype(float)
output = conv2d(
    inputs, filters, input_shape=(1, 1, 5, 5), filter_shape=(1, 1, 3, 3),
    border_mode='valid', subsample=(1, 1))

f = function([inputs, filters], output)

out_feature_map = f(input_feature_map,filter_map)
```

![](2017-05-12_194346.png)

这两个工具包具体的计算过程还没有弄明白，不知道为什么计算结果不一样。
