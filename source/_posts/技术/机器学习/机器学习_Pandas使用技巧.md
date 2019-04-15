---
title: 深度学习_Pandas使用技巧
toc: true

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

### 对DataFrame的某一列进行one-hot编码

```python
from sklearn.preprocessing import OneHotEncoder

# name 列的名称
def one_hot_colum(small_data,name):

    enc = OneHotEncoder()
    data = small_data[name].reshape(len(small_data[name]),1)
    enc.fit(data)
    transformed_data = enc.transform(data).toarray()
    small_data.pop(name)
    for i in range(transformed_data.shape[1]):
        small_data.insert(small_data.shape[1],name+str(i),transformed_data[:,i])

    return small_data
```

### 对DataFrame行进行切分和过滤

下面划分的数据集是腾讯高校算法大赛第一届比赛的数据，实现了按照天为单位划分数据集。总体的思路是使用`isin()`生成`mask`,使用`mask`筛选数据。

```python
def split_data(data,train_data,test_data):
    """
    按照天划分数据集
    :data, DataFrame 类型的数据
    :train_data, 训练数据集，[17,18,19]
    :test_data,测试数据集,[30]
    """

    times = np.unique( train['clickTime'] )
    day = []
    for i in range(14):
        day.append(times[24*60*i:24*60*(i+1)])
    train_data = [i-17 for i in train_data ]
    test_data = [i-17 for i in test_data ]
    mask_train = np.array([False]*train.shape[0])
    for i in train_data:
        mask = train['clickTime'].isin(day[i])
        mask_train |= mask

    mask_test = np.array([False]*train.shape[0])
    for i in test_data:
        mask = train['clickTime'].isin(day[i])
        mask_test |= mask

    return data[mask_train],data[mask_test]
```

### 两个list生成DataFrame并按照某个list排序

```python
aa = pd.DataFrame({'feature':col,'importance':importance})
bb=  aa.sort_values(by='importance')
```

### DataFrame按照某一列的关键字合并

```python
train = pd.merge(dfTrain, dfAd, on="creativeID")
data = pd.merge(data, dfCvr, how="left", on="keyid")
```

### DataFrame随机选择n个样本

```python
dataframe.sample(n=selected_sample_num)
```
