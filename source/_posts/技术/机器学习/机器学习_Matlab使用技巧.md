---
title: 机器学习_Matlab使用技巧
toc: true

tags:
  - matlab
date: 2017-05-04 21:20:58
---

记录一些Matlab常用的操作，以便日后查阅。

<!--more-->

## 发出声音

```Matlab
sp=actxserver('SAPI.SpVoice');
sp.Speak('第一个被试处理完了！')
```

## 计时

```Matlab
tic; % 开始计时
....
toc % 停止计时
```

## 输出重定向到文件

```Matlab
diary 文件路径   % 输出重定向到文件
...
diary off; % 从diary 到diary off 之间的输出都重定向到文件中去了。
```

## 颜色控制

**生成颜色**

Matlab中可以很方便的生成各种不同的颜色，在Matlab中叫做颜色映像。
![](QQ截图20170504215111.png)
例如想要获得 从 蓝色 到 红色 渐变的 100 中颜色，可以输入：
`a = jet(100)`, a会是一个100 * 3的矩阵， 值都在0 -- 1 之间，0表示黑色，1表示白色。 3 个列分别表示R， G ， B 三个通道的取值。

不同的颜色映像：
![](2017-05-04_222428.png)
生成的代码：

```Matlab
color_num = 64;
M = hsv(color_num);
colormap(M);
colorbar();
%pcolor(M);

M = hot(color_num);
colormap(M);
colorbar();

M = cool(color_num);
colormap(M);
colorbar();

M = pink(color_num);
colormap(M);
colorbar();

M = bone(color_num);
colormap(M);
colorbar();

M = jet(color_num);
colormap(M);
colorbar();

M = copper(color_num);
colormap(M);
colorbar();

M = prism(color_num);
colormap(M);
colorbar();

M = flag(color_num);
colormap(M);
colorbar();

M = gray(color_num);
colormap(M);
colorbar();

```
**设置颜色**

使用`colormap(M)`指定画图的时候使用的颜色映像，`M`是上面生成的n*3的矩阵。

**显示colorbar**

函数`colorbar();`可以显示当前的colorbar.

不同的jet(n)得到的颜色：

![](2017-05-04_220731.png)

不同的HSV(n)得到的颜色：

![](2017-05-04_221208.png)

**颜色矩阵**

函数`pcolor(A)`,可以把矩阵A以用颜色代表数值大小的方式显示出来。该函数默认情况下不使用最后一行和最后一列，所以要完整的打印出整个矩阵的颜色，在使用之前要增加一行，增加一列。
 ```Matlab
 % 准备数据
 map=2*(rand(4)-0.5);

 % 增加一行和一列
 clear add_cow;
 clear add_col;
 maxV = max(max(map));
 minV = min(min(map));
 va = (maxV + minV) /2;
 add_cow(1,[1:size(map,2)]) = va;
 map = [map;add_cow];
 add_col([1:(size(map,1))],1) = va;
 map = [map,add_col];
 % 打印
 M = jet(color_num);
 colormap(M);
 pcolor(map);
 ```
不同的颜色映象打印出来的矩阵：
![](2017-05-04_230147.png)

## 读写文件

**读取Excel**

```Matlab
file = 'D:\FMRI_ROOT\YIYU\features_dcm_origin.xls'
data = xlsread(file,'Sheet1');
```

**写入Excel**

```Matlab
xlswrite(file,data,'Sheet2');
```
写入的时候需要确保文件不被其他资源占用，否则可能出现写入失败的情况。

**写文本文件**

```matlab
fid=fopen('D:\colors.txt','w+');
fprintf(fid,'写入的内容');
fclose(fid);
```

## 随机生成数字

```Matlab
rand(n)  % n*n的矩阵
rand(m,n)  % m*n的矩阵
```
