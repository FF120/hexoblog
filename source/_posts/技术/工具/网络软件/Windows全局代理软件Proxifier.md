---
title: Windows全局代理软件Proxifier
toc: true

tags:
  - 网络
date: 2016-06-12 18:28:28
---
Proxifier 是一款功能非常强大的socks5代理服务器，可以让不支持通过代理服务器工作的网络程序能通过HTTPS或SOCKS代理或代理链。支持64位系统，支持Xp，Vista，Win7，支持socks4，socks5，http代理协议，支持TCP，UDP协议，可以指定端口，指定IP，指定程序等运行模式，兼容性非常好。

<!-- more -->

## 安装

[Proxifier](http://www.proxifier.com/)

## 配置

只需要配置代理服务器地址和端口号，Proxifier就可以默认为所有应用程序提供代理
`Profile->Proxy Servers`打开代理服务器配置窗口
![proxy_server](proxy_server.png)
`Address` `port`填代理服务器的地址和端口号，`Protocol`选择`SOCKS Version 5`,然后`ok`

## 设置代理规则

实际使用的时候，可能某个应用程序不想使用代理，例如`utorrent`,可能某些网站不想使用代理，例如学校内网的网站，PT站点等无需认证就能上的站点，可能有的网站需要不同的代理，例如访问国外网站需要一个能翻墙的代理，这些都可以通过配置代理规则来实现。
`profile->Proxification Rules`打开代理规则配置窗口
![proxy_rules](proxy_rules.png)hexo
可以按照程序，目标网站，端口三种类型配置代理规则，配置完成后确定即可启用。
