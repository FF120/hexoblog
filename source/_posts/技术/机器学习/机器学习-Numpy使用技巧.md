---
title: 机器学习_Numpy使用技巧
toc: true

tags:
  - python
  - numpy
date: 2017-07-23 09:50:10
---

NumPy的主要对象是齐次多维数组。表由相同类型的元素组成（通常为数字），由一个正整数元组索引。在NumPy中维数被称为轴，轴数称为秩。例如，一个三维空间中点的坐标[1,2,1]是一个秩为1的数组，因为其轴数为1，轴的长度为3。在下图中，数组的秩为2（2维），第二维的长度为3。
```
[[ 1., 0., 0.],
 [ 0., 1., 2.]]
```

<!-- more -->

## 随机打乱数据

```python
import numpy as np
random_y = np.random.permutation(y)
```
