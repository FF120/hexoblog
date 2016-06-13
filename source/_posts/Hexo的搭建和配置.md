---
title: Hexo的搭建和配置
toc: true
categories:
  - 工具
tags:
  - Hexo
date: 2016-06-13 14:22:06
---
## 添加RSS订阅功能
>安装
``` bash
npm isntall hexo-generator-feed --save
```
>配置
```bash
在博客配置文件 _config.yml 中添加
#添加RSS订阅
feed:
	type: atom
	path: atom.xml
	limit: 20
```
在主题配置文件中 _config.yml 中添加
```bash
rss: /atom.xml
```
