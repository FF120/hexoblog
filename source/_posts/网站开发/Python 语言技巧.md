---
title: Python 语言技巧
toc: true
categories:
  - 编程语言
tags:
  - python
date: 2016-06-19 12:58:08
---

记录python使用过程中的一些代码片段，目的是以后忘记的时候可以随时方便的查询到。

<!-- more -->

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
