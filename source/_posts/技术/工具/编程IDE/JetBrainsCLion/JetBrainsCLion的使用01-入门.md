---
title: JetBrainsCLion的使用01-入门
toc: true

tags:
  - cLion
  - c++
date: 2017-05-04 18:07:21
---

c++的IDE有很多，微软的Visual Studio, 跨平台的codeblocks, 全能的eclipse, 还有就是jetbrains的Clion。 这些我都用过，感觉用着最自然的，花费学习时间最少的就是Clion了，他的调试功能也是非常好用，变量的显示很直观。

<!-- more -->

![](2017-05-04_181337.png)

## 安装CLion

这个软件不是免费的，但是破解方法已经出了，所以可以从官方网站上下载，适用30天，到期再破解就可以了。下载地址：https://www.jetbrains.com/clion/

## 配置编译器

`file->setting->Build,Execution,Deployment`选择`Toolchains`,可以看到右侧支持两种C++编译器，`MinGW`和`Cygwin`, 如果你的电脑上已经安装了任何一种，直接指定该软件的根目录就可以了，软件会自动探测相应的编译器，在线面给出版本。
![](QQ截图20170504181921.png)

**安装`MinGW`**

Windows平台下建议安装[TDM-GCC](http://tdm-gcc.tdragon.net/download),提供Windows下的安装包，安装完成之后直接就能使用，而且他提供了管理工具，以后安装和卸载相关的软件包也比较方便，安装的时候选择最简单的包含C++的编译器就可以。

安装完成后把安装的目录填在上面配置的位置就可以了。

**安装`cygwin`**

下载[cygwin](https://cygwin.com/install.html)按照指示安装即可。

## 字体和外观的调整

`jetbrains`系列的软件都不支持使用`Ctrl+鼠标滚轮`的方式调整字体的大小，这点不太方便。默认的字体和背景可能不太舒服，调整字体和外观的选项都在`file->setting->Editor`中。


## 调试

调试非常方便，只要在左侧单击，出现红色的小圆点，代表断点，然后不要点RUN,而是点DEBUG，就进入了调试模式，程序会运行到断点处等待用户的操作。
![](QQ截图20170504183933.png)
