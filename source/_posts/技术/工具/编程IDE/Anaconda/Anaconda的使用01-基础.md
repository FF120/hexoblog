---
title: Anaconda的使用01-基础
toc: true

tags:
  - python
  - anaconda
date: 2017-04-18 20:11:12
---
[anaconda](https://www.continuum.io/downloads)是一个python的集成环境，自带了许多常用的python包，所以安装它是学习python最简便的方法。anaconda提供conda命令，可以创建多个相互隔离的不同的python工作环境，十分方便。
<!--more-->
## 安装
下载python3.6版本的安装包，直接双击安装即可，所有选项选择默认就行。软件安装好之后，Windows的命令窗口应该可以直接执行`python`,`ipython`,`conda`,`pip`等命令。

## conda
- 查看当前安装的环境 `conda info -e`
- 创建一个名称为`python35`的环境： ` conda create --name python35 python=3.5`
- 删除这个环境： `conda remove -n python35 --all`
- 激活某个环境： `activate python35`
- 退出某个环境：`deactivate`
- 复制某个环境：`conda create -n python35_copy --clone python35`
- 查看conda的版本:`conda --version`
- 更新conda： `conda update conda`
- 查看当前环境中安装的包和版本号： `conda list`
- 一次安装anaconda集成包：`conda install anaconda`
- 更新anaconda的版本： `conda update anaconda`
- 更改conda的镜像：
```shell
# 添加Anaconda的TUNA镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# TUNA的help中镜像地址加有引号，需要去掉

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
```

## 配置python2.7 和python 3.5两个工作环境
1. `conda create --name python35 python=3.5`
2. `activate python35`
3. `conda install anaconda`
4. `spyder`


这样就打开了python3.5的spyder工作环境。`conda install anaconda`是安装anaconda所有的包，如果用不到这么多，可以不必执行这一步，因为安装太多的包比较耗时。

1. `conda create --name python27 python=2.7`
2. `activate python27`
3. `conda install anaconda`
4. `spyder`


这样就打开了python3.5的spyder工作环境。
