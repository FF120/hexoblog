---
title: GIT的使用01-基本功能
toc: true

tags:
  - git
date: 2017-05-04 18:41:45
---

版本控制工具主要有SVN和git, 目前git已经全面超越了SVN, 成为最优秀的版本管理工具。一切文本文档都可以用git来管理，使用git能够以最小的存储代价，追踪记录每一次对文件的修改，并且能够在需要的时候，恢复到任意一个版本。 除此之外，多人协作也是git的一个重要应用，它可以方便的完成分支和合并，提高团队合作的效率。

<!-- more -->

目前基于git的开放源代码托管平台国内主要是(coding.net)[coding.net],免费提供私有仓库。国外主要是[github](github.com),两者的功能基本一样，都提供代码管理和Page服务，所以两者都可以用来托管静态博客。

目前应用git较多的领域是一个是编程领域，包括代码版本的管理，文档的管理；另一个就是图书出版领域，有很多写书的人就是利用git管理自己的内容的。有不少工具提供了将文本格式格式化成书籍的工具，最常见的一种是`markdown`.PREVIEW

## 安装

[git](https://git-scm.com/)提供了Windows的安装包，所以直接下载安装就可以了。初学git,最好使用命令，放弃图形界面，这样能更深入的理解内部的原理，这对以后复杂的版本管理，分支合并等内容有好处。

## 使用

### 克隆别人的仓库

```bash
git clone 仓库地址
```

如果是公开仓库，直接克隆成功，如果是私有仓库，按照提示输入用户名和密码。

### 修改之后提交到远程

```bash
git add -A  #添加所有新添加的文件
git commit -a -m "提交说明"  #提交修改到本地git仓库
git push origin master # 提交本地仓库到远程的master分支
```

### 保持和远程仓库一致

```bash
git pull
```

pull之后可能会出现冲突的情况，这个时候git会提示你有哪些文件冲突了，你需要子集修改冲突的文件，然后重新提交。

### 记住密码


修改.git目录中的config将其中的`[remote "origin"]`修改为

```
url = https://[username]:[password]@github.com/...
```

<!-- more -->

![example for rememner git password](remember_password.png)

### 进一步学习

git的几个简单的命令足够日常使用，如果项进一步学习，可以参考一下资料：
[git简明指南](http://rogerdudler.github.io/git-guide/index.zh.html)
[ProGit](https://git-scm.com/book/zh/v2)
