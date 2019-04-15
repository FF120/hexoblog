---
title: 机器学习_Scipy使用技巧
toc: true

tags:
  - ML,统计
date: 2017-06-15 19:38:53
---

scipy是一个统计工具包，包含各种各种的统计方法。本文主要记录各种统计计算需要调用的函数名称和函数的调用格式。

<!--more-->

## 单样本T检验

```python
from scipy import stats
pvalues = stats.ttest_1samp(dffeatures,mean=0.0,axis=0)
```

## 双样本T检验

```python
from scipy import stats
pvalues = stats.ttest_ind(group1,group2)
```
