---
title: 深度学习_限制波尔茨曼向量机(RBM)
toc: true

tags:
  - RBM
  - 限制波尔茨曼机
date: 2017-05-10 12:34:57
---
本文主要介绍RBM相关的知识。包括能量函数，概率的观点，网络的结构等等。
<!--more-->
先来看看没有限制的波尔茨曼机是什么样子的。

![](2017-05-10_123815.png)

原始的波尔茨曼机虽然有很多有趣的特性，但是由于连接过于复杂，很难实际应用（或者是有效的方法还没有被提出来，就像当年深度网络缺乏有效的训练方法一样）。一种限制连接形式的波尔茨曼机因为其简单的结构和有趣的性质得到了广泛的应用。看看它长什么样子。

![](2017-05-10_123931.png)

这就是RBM，很像一个两层的全连接的神经网络。
我们用$v_i$来表示可视层的神经元,用$h_j$来表示隐藏层的神经元,用 $w_{ij}$ 来表示两层之间的权重，用 $a_i$ 来表示可视层的偏置，用 $b_j$ 表示隐藏层的偏置,用 $(v,h)$ 表示一个RBM，那么我们可以定义一些经常要使用到的量.

RBM的能量E定义为：

$$
E(v,h)=-\sum_i{a_i v_i}-\sum_j{b_j h_j}-\sum_i\sum_j{h_j w_{ij} v_i}
$$

在一般的玻尔兹曼机中，隐层和可见层之间的联合概率分布由能量函数给出:

$$
P(v,h) = \dfrac 1 Z e^{-E(v,h)}
$$

其中， $Z$ 为配分函数，定义为在节点的所有可能取值下$ e^{-E(v,h)} $的和（亦即使得概率分布和为1的归一化常数）.
类似地，可见层取值的边缘分布可通过对所有隐层配置求和得到:
$$
P(v)=\dfrac 1 {Z} \sum_h{e^{-E(v,h)}}
$$
可见层的配置v对于隐层配置h的条件概率如下:
$$
P(v|h)=\prod_{i=1}^m P(v_i|h)
$$
h对于v的条件概率为:
$$
P(h|v)=\prod_{j=1}^n P(h_j|v)
$$

单个节点的激活概率为

$$
P(h_j=1|v)=\sigma(b_j+\sum_{i=1}^m {w_{ij} v_i})
$$

$$
P(v_i=1|h)=\sigma(a_i+\sum_{j=1}^n w_{ij}h_j)
$$

式中的$\sigma$表示Logistic函数，：
$$
P(t)= \dfrac 1 {1+e^{-t}}
$$

受限玻尔兹曼机的**训练目标**是针对某一训练集 ${\displaystyle V}$，最大化概率的乘积。其中， ${\displaystyle V} $被视为一矩阵，每个行向量作为一个可见单元向量 ${\displaystyle v}$:
$$
\arg\max_W \prod_{v \in V} P(v)
$$

等价的，可以最大化它的对数函数：
$$
\arg\max_W \mathbb{E} \left[\sum_{v \in V} \log P (v)\right]
$$

训练受限玻尔兹曼机，即最优化权重矩阵 ${\displaystyle W}$ .

波尔茨曼机也可以看作是 马尔科夫随机场的一种特殊情况。

RBM一个基本的训练方法叫做Contrastive Divergence，它的具体过程如下：

输入： 一个N行的矩阵$x_0$，每一行对应一个可视的神经元; 隐层单元的个数m,学习率$\epsilon$, 最大训练周期T。

输出：权重$W$, 可视层的偏置$a$, 隐藏层的偏置$b$

初始化： 权重$W$,偏置$a$,$b$随机初始化为较小的数值,令可视层的第一个状态$v_1 = x_0$


参数更新规则：

$W = W + \epsilon*( P(h_1=1|v_1 )v_1^T - P(h_2=1|v_2)v_2^T))$

$a = a +\epsilon*(v_1 - v_2)$

$b = b + \epsilon*(P(h_1=1|v_1)-P(h_2=1|v_2))$

下面用Python实现其中的计算步骤，这样能够对算法的每一步有一个比较详细的了解。

```python
from __future__ import print_function
import numpy as np

class RBM:

  def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0 and standard deviation 0.1.
    self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.
    """
    # 也是可视层的神经元的数量
    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    # data的第一列用来表示偏置，这里全部设置成1
    data = np.insert(data, 0, 1, axis = 1)
    # max_epochs 就是最大的训练周期T
    for epoch in range(max_epochs):
      # Clamp to the data and sample from the hidden units.
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.

    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1

    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states

  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
  r = RBM(num_visible = 6, num_hidden = 2)
  training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
  r.train(training_data, max_epochs = 50)
  print(r.weights)
  user = np.array([[0,0,0,1,1,0]])
  print(r.run_visible(user))


```

**参考文献**

1. 介绍+Python源代码：<http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/>
2. 介绍+Python源代码：<http://imonad.com/rbm/restricted-boltzmann-machine/>
