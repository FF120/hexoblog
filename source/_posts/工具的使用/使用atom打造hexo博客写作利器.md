---
title: 使用atom打造hexo博客写作利器
toc: true
categories:
  - 工具
tags:
  - hexo
  - atom
date: 2017-04-12 10:51:29
---
自从开始使用hexo写博客以来，一直在寻找一款合适的编辑器，至少实现以下功能：
> - markdown语法的着色
- markdown中代码块的着色和编程提示
- 实时预览
- 根据标题生成目录
- 在编辑器中执行`hexo new, hexo d,hexo g,hexo clean`等命令
- 在编辑器中执行`git`相关的操作
- 可视化的界面显示文件修改的异同
- 导出markdown为带书签的PDF，样式可以自定义

直到遇到atom，上述功能轻松就实现了，还有许多其他扩展的功能可以使用。
<!-- more -->
## 安装atom
下载[atom](https://atom.io/), 安装完成之后默认支持markdown功能，支持实时预览。

## 支持hexo命令的配置
`Ctrl+,`打开配置页面，在install中搜索hexo相关的扩展，安装之后即可使用。atom的命令窗口与sublimeText一样，是`Ctrl+Shift+P`

## 增加目录功能
在扩展安装页面搜索markdown相关的扩展，把需要的都安装上。可以实现添加目录，导出PDF等各种功能扩展。

## 扩展git操作
安装git-plus扩展，可以在命令窗口执行相关的操作。

## 增加miniView插件
安装miniView扩展，可以看到整个页面的缩略图。
