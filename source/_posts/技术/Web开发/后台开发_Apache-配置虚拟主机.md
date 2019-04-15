---
title: Apache 配置虚拟主机
toc: true

tags:
  - apache
  - php
date: 2016-06-12 16:44:58
---
Apache支持多个虚拟主机的设置，可以实现在一个IP上部署多个网站的效果。例如，在本机实现每个域名访问一个对应的网站，Apache应该如下配置。

<!-- more -->

## 修改`http.conf`文件
``` bash
<Directory "E:/WebRoot">
    AllowOverride All
    Require all granted
</Directory>
```
上面的代码将文件夹`W:/WebRoot`设置成可访问

``` bash
Include "conf/extra/httpd-vhost.conf"
```
在主配置文件中包含虚拟主机配置文件，虚拟主机的信息在`httpd-vhost.conf`中配置

## 修改`httpd-vhost.conf`文件
``` bash
<VirtualHost *:80>
    ##ServerAdmin webmaster@dummy-host.example.com
    DocumentRoot "E:/WebRoot"
    ServerName localhost
    ServerAlias localhost
    ErrorLog "logs/blog.com-error.log"
    CustomLog "logs/blog.com-access.log" common
</VirtualHost>
```
第一个节点是默认节点，所有没有匹配到的域名都会走这个节点，所以一般把这里设置成Web根目录，域名就是localhost,这样以后可以用localhost访问到这个web根目录的内容。

``` bash
<VirtualHost *:80>
    ##ServerAdmin webmaster@dummy-host.example.com
    DocumentRoot "E:/WebRoot/laravel-5-blog/public/"
    ServerName my.blog.com
    ServerAlias my.blog.com
    ErrorLog "logs/blog.com-error.log"
    CustomLog "logs/blog.com-access.log" common
</VirtualHost>
```
像上面这样配置，当在浏览器输入my.blog.com的时候，就会访问到E:/WebRoot/laravel-5-blog/public/下，这里需要注意的是，一定要在httpd.conf里面设置了该文件夹的可访问权限，否则是会拒绝访问的。

``` bash
<VirtualHost *:80>
    ServerAdmin webmaster@dummy-host2.example.com
    DocumentRoot "E:/WebRoot/wordpress/"
    ServerName my.wordpress.com
    ErrorLog "logs/wordpress.com-error.log"
    CustomLog "logs/wordpress.com-access.log" common
</VirtualHost>
```
上面增加了一个域名为my.wordpress.com的站点。

## 修改系统`hosts`文件
``` bash
127.0.0.1       my.blog.com
127.0.0.1       my.wordpress.com
```
然后就可以在浏览器使用对应的域名访问对应的网站了。这样设置的好处是对于那些路由有特殊要求的框架适应性比较好，不会出现资源找不到情况。另外，如果想在一台服务器上安装多个网站，这也是一个可行的方法。
