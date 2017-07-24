---
title: 机器学习_Matplolib使用技巧
toc: true
categories:
  - 编程语言
tags:
  - python
  - matplotlib
date: 2017-07-23 09:52:18
---
Matplotlib 可能是 Python 2D-绘图领域使用最广泛的套件。它能让使用者很轻松地将数据图形化，并且提供多样化的输出格式。这里将会探索 matplotlib 的常见用法。

<!-- more -->

## IPython 以及 pylab 模式

IPython 是 Python 的一个增强版本。它在下列方面有所增强：命名输入输出、使用系统命令（shell commands）、排错（debug）能力。我们在命令行终端给 IPython 加上参数 -pylab （0.12 以后的版本是 --pylab）之后，就可以像 Matlab 或者 Mathematica 那样以交互的方式绘图。

**pylab**

pylab 是 matplotlib 面向对象绘图库的一个接口。它的语法和 Matlab 十分相近。也就是说，它主要的绘图命令和 Matlab 对应的命令有相似的参数。

## 初级绘制

这一节中，我们将从简到繁：先尝试用默认配置在同一张图上绘制正弦和余弦函数图像，然后逐步美化它。

第一步，是取得正弦函数和余弦函数的值：

```python
import numpy as np
from pylab import *

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)
```

X 是一个 numpy 数组，包含了从 −π−π 到 +π+π 等间隔的 256 个值。C 和 S 则分别是这 256 个值对应的余弦和正弦函数值组成的 numpy 数组。


[原文](https://liam0205.me/2014/09/11/matplotlib-tutorial-zh-cn/)
