---
title: Bootsharp学习笔记
toc: true

tags:
  - bootsharp
date: 2016-06-30 22:51:32
---
![bootstrap](bootstrap.png)
<!-- more -->
# 简介

Bootstrap 是一个用于快速开发 Web 应用程序和网站的前端框架。Bootstrap 是基于 HTML、CSS、JAVASCRIPT 的。
Bootstrap，来自 Twitter，是目前最受欢迎的前端框架。Bootstrap 是基于 HTML、CSS、JAVASCRIPT 的，它简洁灵活，使得 Web 开发更加快捷。

# 安装

从 http://getbootstrap.com/ 上下载 Bootstrap 的最新版本。
- •	`Download Bootstrap`：下载 Bootstrap。点击该按钮，您可以下载 Bootstrap CSS、JavaScript 和字体的预编译的压缩版本。不包含文档和最初的源代码文件。
- •	`Download Source`：下载源代码。点击该按钮，您可以直接从 from 上得到最新的 Bootstrap LESS 和 JavaScript 源代码。

# 使用

```html
<!DOCTYPE html>
<html>
<head>
   <title>在线尝试 Bootstrap 实例</title>
   <link href="/bootstrap/css/bootstrap.min.css" rel="stylesheet">
   <script src="/scripts/jquery.min.js"></script>
   <script src="/bootstrap/js/bootstrap.min.js"></script>
</head>
<body>
     <h1>Hello, world!</h1>
	 </body>
</html>
```
解释说明：
1. Bootstrap 使用了一些 HTML5 元素和 CSS 属性。为了让这些正常工作，您需要使用 HTML5 文档类型（Doctype）。因此，请在使用 Bootstrap 项目的开头包含下面的代码段。
`<!DOCTYPE html>`
2. bootstrap 不需要任何安装部署 和 开发环境，只需要包含以下几个文件：
```html
   <link href="/bootstrap/css/bootstrap.min.css" rel="stylesheet">
   <script src="/scripts/jquery.min.js"></script>
   <script src="/bootstrap/js/bootstrap.min.js"></script>
```
就可以使用bootstrap的各种效果了。
3. Bootstrap 3 的设计目标是移动设备优先，然后才是桌面设备。
为了让 Bootstrap 开发的网站对移动设备友好，确保适当的绘制和触屏缩放，需要在网页的 head 之中添加 viewport meta 标签，如下所示：
`<meta name="viewport" content="width=device-width, initial-scale=1.0">`
	注释：
	- `initial-scale=1.0` 确保网页加载时，以 1:1 的比例呈现，不会有任何的缩放。
    - 在移动设备浏览器上，通过为 `viewport meta` 标签添加 `user-scalable=no` 可以禁用其缩放（zooming）功能。通常情况下，`maximum-scale=1.0` 与 `user-scalable=no` 一起使用。这样禁用缩放功能后，用户只能滚动屏幕，就能让您的网站看上去更像原生应用的感觉。
	- 移动设备通常使用的meta标签：`<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">`

## 响应式图像
```html
<img src="..." class="img-responsive" alt="Responsive image">
```
通过添加为`img`标签添加`class="img-responsive"`可以使图像的响应式布局更友好。
```html
.img-responsive {
  display: inline-block;
  height: auto;
  max-width: 100%;
}
```
Bootstrap 的css中的`img-responsive`是这样写的，保证了响应式布局。从这里可以看出，bootstrap做的事情其实就是定义了一些css类的样式，你只要包含这些文件，就能直接使用这些样式，而这些样式是经过精心设计的，保证了满足响应式布局的需要。

## 网格系统

Bootstrap 响应式的网格系统可以根据设备大小进行灵活的适应，保证内容的显示效果。
Bootstrap的网格系统将屏幕分为12列，在定义每一列的时候，可以指定一个数字，代表占用几列，bootstrap 自动用相应的比例显示他们。
`<div class="container">...</div>` 元素被添加，确保居中和最大宽度。
一旦添加了容器，接下来您需要考虑以行为单位。添加` <div class="row">...</div>`，并在行内添加列 `<div class="col-md-6"></div>`。
有两个列，每个列由 6 个单元组成，即 6+6=12。
在大型设备显示时修改布局：
```html
<div class="col-md-6 col-lg-4">....</div>
<div class="col-md-6 col-lg-8">....</div>
```

这样，当在middle设备上显示时，会显示col-md-6这个Class，在large设备上显示时，会显示col-g-4这个Class。
所以，在中型设备上，两列宽度是1：1，在大型设备上，两列宽度是2：3.
```html
<div class="col-sm-3 col-md-6 col-lg-4">....</div>
<div class="col-sm-9 col-md-6 col-lg-8">....</div>
```
同理，如果这样设置，就能实现在小型、中型、大型设备上以不同的比例显示列。

响应式的列重置：
```html
<div class="col-xs-6 col-sm-3"
<div class="col-xs-6 col-sm-3"
<div class="clearfix visible-xs"></div>
<div class="col-xs-6 col-sm-3"
<div class="col-xs-6 col-sm-3"
```
当屏幕足够大时，四列同宽（3:3:3:3），显示在同一行；当不够宽时，分两行显示，第一行显示前两列，第二行显示后两列。宽度是（6：6）
偏移列：
` <div class="col-xs-6 col-md-offset-3"`偏移3列，占据6列，右边还剩3列，所以这句的效果就是 居中。
嵌套：
```html
<div class="row">
       <div class="col-md-3"
       <div class="col-md-9"
	    <div class="row">
		   <div class="col-md-6"
		   <div class="col-md-6"
	    <div class="row">
		   <div class="col-md-6"
		    <div class="col-md-6"
```
这样就能嵌套出这样的一个表格：
![1](1.png)

列排序：
1. `<div class="col-md-4 col-md-push-8" `
2. `<div class="col-md-8 col-md-pull-4" `
第一列占四列，第二列占8列，这样显示出来是左变窄，右边宽的一个布局。我们可以通过col-md-push-8 和col-md-pull-4 改变这两列的显示顺序，形成左边宽，右边窄的布局。Push-8 的意思就是将占8行的列提前，pull-4的意思就是将占4行的列退后。

## 响应式表格
