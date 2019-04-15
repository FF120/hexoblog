---
title: Hexo的使用01-搭建
toc: true

tags:
  - hexo
date: 2016-06-13 14:22:06
---

hexo是一个快速、简洁且高效的博客框架，它使用渲染引擎渲染markdown文本，生成静态网页。任何可以托管静态网页的空间都可以部署hexo博客。目前流行的组合是`hexo + git + github`. 可以实现像管理代码的版本那样管理自己的文档。

<!-- more -->

## 安装

Hexo的安装和部署十分简单，它是基于`Node.js`的，所以首先确保安装了`Node.js`.在系统环境中`npm`命令是可以使用的。

```js
npm install hexo-cli -g
hexo init blog
cd blog
npm install
hexo server
```

## 配置

Hexo分成了两个层级，Hexo应用层面的配置，配置文件在博客根目录的`_config.yml`,另外一个是主题配置文件，在`themes\[主题名]\_config.yml`中。主题的配置参考你使用的主题的相关的文档配置。
网站的配置参考：https://hexo.io/zh-cn/docs/configuration.html

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

## nexT主题的配置

[nexT](http://theme-next.iissnan.com/getting-started.html)是一个界面简洁，干净的主题，很流行。

### 修改网页背景颜色

**主题颜色**

打开hexo/themes/next/source/css/_variables/base.styl找到Colors代码段，如下：
```
// Colors
// colors for use across theme.
// --------------------------------------------------
  $whitesmoke   = #f5f5f5
  $gainsboro    = #eee  //这个是边栏头像外框的颜色，
  $gray-lighter = #ddd  //文章中插入图片边框颜色
  $grey-light   = #ccc  //文章之间分割线、下划线颜色
  $grey         = #bbb  //页面选中圆点颜色
  $grey-dark    = #999
  $grey-dim     = #666 //侧边栏目录字体颜色
  $black-light  = #555 //修改文章字体颜色
  $black-dim    = #333
  $black-deep   = #495a80  //修改主题的颜色，这里我已经改成老蓝色了。
  $red          = #ff2a2a
  $blue-bright  = #87daff
  $blue         = #0684bd
  $blue-deep    = #262a30
  $orange       = #F39D01 //浏览文章时，目录选中的颜色
```

**主题背景**

打开hexo/themes/next/source/css/_schemes/Pisces/index.styl(Pisces为NexT提供的三种主题之一，根据使用的主题选择）修改body{}内的值，如下：
背景颜色直接更改即可：body { background: #F0F0F0; }
添加背景：body { background: url（'/images/background.jpg'); }(将背景图片放到hexo/source/images中即可。

**内容背景**

修改博客背景颜色
/themes/next/source/css/_schemes/Pisces/_layout.styl
```
.content-wrap {
  background: #222222;
}
```

### 添加disqus评论支持

https://disqus.com 是一个国际上使用最广泛的评论系统，可以方便的安装在任何网站之上。nexT主题自带了DISQUS评论代码，只需要在配置文件中填上子集的short_name就可以了。
**disqus帐号的申请和使用**
首先到disqus的官方网站申请一个帐号，登陆。
![](2017-05-02_195454.png)
选择`add disqus to site`, 在网页的最下端选择`get start`
![](2017-05-02_195700.png)
选择第二个，我要简历一个网站，
![](2017-05-02_195752.png)
填写网站的相关信息，其中`Website Name`是唯一的，也就是NexT主题中填入的shor_name.

设置成功之后的样子：
![](2017-05-02_195947.png)


## maupassant主题配置

另外一个比较简洁的主题，风格类似nexT.
https://github.com/tufu9441/maupassant-hexo
