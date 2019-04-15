---
title: JQuery EasyUI 学习笔记
toc: true

tags:
  - jquery
date: 2016-06-30 22:44:37
---
![easyUI0.png](easyUI0.png)
<!-- more -->

# 安装

下载[jquery EasyUI](http://www.jeasyui.com/index.php) 安装包
![easyUI](easyUI.png)
解压缩之后的目录结构为
![structure](structure.png)
要使用easyui只需要在HTML页面包含
```html
<link rel="stylesheet" type="text/css" href="themes/default/easyui.css">
<script type="text/javascript" src="jquery.min.js"></script>
<script type="text/javascript" src="jquery.easyui.min.js"></script>
```
指定下载的文件的位置就可以直接使用里面定义好的控件了。

# 主要内容
1. 网格系统
	表头：
```html
<thead>
<tr>
<td>第一列</td>  <td>第二列</td>  <td>第三列</td>
</tr>
</thead>
```
	表体:
```html
<tbody>
<tr>
<td>第一列</td>  <td>第二列</td>  <td>第三列</td>
</tr>
</tbody>
```
