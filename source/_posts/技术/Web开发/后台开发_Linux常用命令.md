---
title: Linux常用命令
toc: true

tags:
  - linux
date: 2016-06-11 20:08:53
---
介绍Linux常用命令的主要用法
<!-- more -->
``` bash
- 查找某个软件包是否安装：
yum list installed|grep mcrypt
- 查看某个命令的位置
whereis ls
- 检查某个端口是否打开
lsof -i :port
- 查看某个运行的进程
ps -ef|grep httpd
- 查找某个进程
pgrep -l ssh

```
