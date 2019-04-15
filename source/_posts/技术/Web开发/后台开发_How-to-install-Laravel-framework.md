---
title: How to install Laravel framework
date: 2016-06-11 12:58:02
toc: true
tags:
- php
- laravel

---

## 安装composer
1. 下载composer的Windows安装包，[ Composer-Setup.exe ](https://getcomposer.org/download/)
2. 运行安装程序，配置`php.exe`所在的位置
![php.exe's path set](set_php_path.jpg)
安装程序会自动从官网下载所需的文件，自动完成安装
![downloading files](downloading_files.jpg)

<!-- more -->


3. 安装完成后，打开`cmd`，键入`composer`，出现如下提示证明安装成功
![composer hint](composer_hint.png)
4. 如果因为没有翻墙安装失败，可以使用国内composer镜像安装
[composer中国镜像](http://pkg.phpcomposer.com/)
5. composer 离线安装方法
- 下载需要的文件
百度云：链接: http://pan.baidu.com/s/1qW0VhCC 密码: 2nc1
![file capture](file_capture.jpg)
- 将文件解压至任意目录，例如`c:\composer\`
- 将`bin`所在的路径添加到系统的`path`变量中
- 重新打开一个`cmd`，键入`composer`测试

## 安装Laravel框架
1. 官方文档
[官网教程](https://laravel.com/docs/5.2#installation)
2. 下载框架安装
百度云： 链接: http://pan.baidu.com/s/1mg84OCo 密码: ev2x
下载完成后，解压文件到Web服务器目录下，文件结构为
![laravel framework structure](laravel_framework_structure.png)
3. 在该目录下运行 composer install ，安装lavarel框架所需要的依赖包
![composer install laravel](composer_install_laravel.png)
安装完成后，会在该文件夹下新增一个vender文件夹，这里面是lavarel框架的依赖包。
vender 的结构
![vender structure](vendor_structure.png)
4. 在项目根目录下键入`php artisan serve`
![php artisan serve](php_artisan_serve.png)
在浏览器输入`localhost:8000`出现如下页面，证明安装成功
![install laravel successful](install_laravel_successful.png)

## 听听音乐
<center> <iframe name="iframe_canvas" src="http://douban.fm/partner/baidu/doubanradio" scrolling="no" frameborder="0" width="400" height="200"></iframe> </center>
