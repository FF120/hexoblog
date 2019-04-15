---
title: 深度学习_Python使用技巧
toc: true

tags:
  - python
date: 2017-06-15 19:47:07
---

记录所有涉及Python的短语句的写法。

<!--more-->

常用的包的引入和别名：

```python
import xlrd
import xlwt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_curve, auc
from sklearn import preprocessing
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif,RFECV
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold,LeavePOut,LeaveOneOut
from sklearn.model_selection import train_test_split
```

## Python

**目录操作**

```python
import os
dirs_and_files = os.listdir(r'd:/')   #
os.chdir(r'd:/')
os.path.join(path1,path2)  # 拼接路径
```

**读写文本文件**

```python
file_object = open(filepath, 'w')
file_object.write(string)
file_object.close()
```

**读写Excel**

```python
import xlrd
data = xlrd.open_workbook(excelfile)
table = data.sheets()[0]          #通过索引顺序获取
table = data.sheet_by_index(0)   #通过索引顺序获取
table = data.sheet_by_name(u'详细信息')#通过名称获取
cellij = table.cell(i,j).value

import xlwt
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('sheet1')
worksheet.write(i, j, label = value)
workbook.save(r'excel.xls')

import pandas as pd
dataframe = pd.read_excel(filepath,sheetname='sheet1',header=None,index_col=None)
dataframe = pd.read_csv(filepath,sheetname='sheet1',header=None,index_col=None)

dataframe.to_excel(filepath,sheet_name='sheet2',header=False,index=False)
dataframe.to_csv(filepath,sheet_name='sheet2',header=False,index=False)

```

### list类型转换成string类型输出

使用python将list类型的数据转换成string类型的。eg: [1,2,3,4,5,6] to 1,2,3,4,5,6

```python
def list_to_str(list):
    str1 = str(list)
    str1 = str1.replace(']','').replace('[','').replace(' ','')
    return str1

for line in open('e:/test_sigmoid222.txt','r'):
    aa =  line.strip('\n')  .split('\t');
    bb = map(int,aa[1].split(','));
    cc = []
    maxValues = max(bb)
    minValues = min(bb)
    for x in bb:
        y = (float)(x-minValues)/(maxValues-minValues)
        y = (int)(y*1000)
        cc.append(y)
    with open('e:/test_sigmoid.txt','a') as of:
        outstr = aa[0]
        outstr = outstr + "\t"
        outstr = outstr + list_to_str(cc)
        outstr = outstr + "\n"
        of.write(outstr)
```

## 输入和输出

### 读取首行是字段名称的CSV数据，或者文本数据

```python
import pandas as pd
# pandas.dataFrame 类型
data = pd.read_csv(file_path)
# numpy  ndarray 类型
data_matrix = data.as_matrix()
```

### 从控制台读取整数

```python
import string
try:
    lists = []
    while True:
        line = raw_input().split()
        lists.append(string.atoi(line[0]))
except EOFError:
    pass


## 或者这样写
(x,y) = (int(x) for x in raw_input().split())
```

### 读取控制台一行字符串

```python
try:
    lineStrings=[]
    lineStrings.append(raw_input())

except EOFError:
    pass

# 或者这样写
import sys
num = sys.stdin.readline()[:-1] # -1 to discard the '\n' in input stream
```

### 每次输出一个字符

```python
import sys
sys.stdout.write('a')
sys.stdout.write(' ')
```

### 倒序输出一个List

```python
print range(10)[::-1]
```

## 格式转换

### 字符串转整形和浮点型

```python
import string
string.atoi()
string.atof()
```

## 数序运算

```python
a = 10
print a*2
print a**2
print a**0.5
print a%10
print a%3
print a/3
```

