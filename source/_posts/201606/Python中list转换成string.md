---
title: Python中list转换成string
toc: true
categories:
  - Python
tags:
  - python
date: 2016-06-19 12:58:08
---
使用python将list类型的数据转换成string类型的。eg: [1,2,3,4,5,6] to 1,2,3,4,5,6
<!-- more -->
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
