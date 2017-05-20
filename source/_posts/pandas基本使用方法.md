---
title: pandas基本使用方法
toc: true
categories:
  - 编程语言
tags:
  - pandas
  - python
date: 2017-05-17 18:29:02
---

pandas 是提供一种类似表格结构的数据结构的Python工具包，使用它可以很方便的完成若干在电子表格中的操作。

<!-- more -->

## 安装

```python
conda install pandas
```

## 数据结构

### 引入

```python
 import pandas as pd
```

### Series

One-dimensional ndarray with axis labels (including time series).

```python
# 创建Series
s1 = pd.Series(5, index=['a'],name='s1')
s2 = pd.Series(['first','second',],index=[0,1],name='s3')
s3 = pd.Series({'a':1,'b':2,'c':3})
# 获取
s1[0] # 按照索引
s3['a'] # 按照键值
s2.index # 获得所有的索引
s2.get('a','empty') # 使用get,不存在的键返回自定义的值
s2[0:2] # 范围截取
s1.name # 获得name属性
s1.rename("different")
```

### DataFrame

a 2-dimensional labeled data structure with columns of potentially different types.

```python
# 创建

d1 = {'one' : ['one','two','third'],
     'two' : [4,5,6]}
# 每个键值一列
df1 = pd.DataFrame(d1)
# list 中是 dict
d2 = [{'a':1,'b':2,'c':3},{'a':1,'b':2,'c':3},{'a':1,'b':2,'c':3}]
df2 = pd.DataFrame(d2,index=['aa','bb','cc'])
d3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
df3 = pd.DataFrame(d3)

# 获取
df1.index  # 行标号
df1.columns # 列标号
df2['a'] # 一列
df2.head() # 显示部分信息


# 修改
del df2['a']
df2.pop('a')  # 删除一列
df2['inserted'] = 'a' # 插入一列
df2['insert2'] = [1,2,3]
df2.insert(0,'between',[1,2,3]) # 指定插入的位置
df2['aa'] #
```
