---
title: Hexo的使用03-迁移
toc: true

tags:
  - hexo
date: 2017-05-05 00:10:05
---

hexo目前还没有全自动的一键迁移功能，所以要移动位置的时候需要重新执行一边流程。把这个流程记录下来，便于以后再次迁移的时候使用。

<!--more-->

## 建站

1. 找一个目录，命令行切换到该目录下。
2. `npm install -g hexo-cli`, 安装主程序。
3. `hexo version`, 查看安装的版本
![](QQ截图20170505001427.png)
4. `hexo init .`, 在当前目录下建立站点，该过程会安装许多相关的文件。
![](QQ截图20170505001745.png)
5. `hexo s`，打开hexo的服务器，然后在浏览器输入`http://localhost:4000/`,查看效果。
![](QQ截图20170505002008.png)

## 安装需要的插件

`npm install hexo-deployer-git --save`
` npm install hexo-generator-feed --save`
`npm install hexo-generator-search --save`

## 更新成以前的配置

主配置文件和整个主题文件夹全部拷贝过来，覆盖原来的文件。

## 拷贝博客数据

把source文件加整个复制过来覆盖。

## 完成

此时本地的环境就搭建好了，只是还没有和远程的仓库连接起来。
`hexo g`生成public目录。
将public部署到远程。

## 解决hexo和latex的冲突问题

https://github.com/hexojs/hexo/issues/524

使用pandoc会出现其他的问题，因为markdown语法会有少许的不同，这个是最令人头疼的事情。
http://shomy.top/2016/10/22/hexo-markdown-mathjax/

适用了N中方法，最后发现，还是修改源码的方法最靠谱，不会有格式上的太大的变化。
