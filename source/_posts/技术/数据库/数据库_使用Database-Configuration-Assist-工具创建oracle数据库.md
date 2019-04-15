---
title: 使用Database Configuration Assist 工具创建oracle数据库
toc: true

tags:
  - oracle
date: 2016-06-12 09:59:52
---
## 启动Database Configuration Assist 工具,在oracle用户下输入dbca，如果出现图形化安装界面就说明启动成功了，如果提示错误，就比较麻烦了，第一个方法是先回到root用户下，输入xhost +，回车，然后再回到oracle用户下，重新输入dbca，如果还打不开图形界面，那么就麻烦了，要重新装一下昨天装的软件，直接输入cd /soft/database/，然后ls，然后./runInstall，重新装一遍，有一个界面是提示出现一个错误和一个警告，把那两个都打上对勾，然后完成安装，之后在oracle中接着输入dbca就Ok了，进入到图形安装界面。

<!-- more -->

## 创建数据库
![create_database](create_database.jpg)
![create_database2](create_database2.jpg)
## 选择一般数据库
![create_database3](create_database3.jpg)
## 给数据库命名要求为：组号+姓名首字母缩写
![create_database4](create_database3.jpg)
## 选择使用EM配置数据库，使用Database Control管理数据库
![create_database5](create_database5.jpg)
## 密码统一使用oracle
![create_database6](create_database6.jpg)
## 选择文件系统
![create_database7](create_database7.jpg)
## 选择数据文件的目录
![create_database8](create_database8.jpg)
## 指定闪回恢复区
![create_database9](create_database9.jpg)
## 选择sample Schema
![create_database10](create_database10.jpg)
## 字符集选择如下
![create_database11](create_database11.jpg)
## database storage 概览
![create_database12](create_database12.jpg)
## 创建数据库，保存为数据库模板，并生成创建脚本
![create_database13](create_database13.jpg)
## 数据库配置概览
![create_database14](create_database14.jpg)
## 模板创建完成及脚本生成
![create_database15_1](create_database15_1.jpg)
![create_database15_2](create_database15_2.jpg)
## 如下创建过程直至退出
![create_database16_1](create_database16_1.jpg)
![create_database16_2](create_database16_2.jpg)
