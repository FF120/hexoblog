---
title: Python速查手册
toc: true
categories:
  - 编程语言
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

## Numpy

**随机打乱数据**

```python
random_y = np.random.permutation(y)
```

## Matplotlib

## Scipy

## Scikit-learn

## seaborn
