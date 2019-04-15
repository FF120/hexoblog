---
title: Hexo的使用02-同步
toc: true

tags:
  - hexo
  - git
date: 2016-06-12 20:02:43
---

Hexo生成的博客在Public文件夹中。我们使用命令`hexo d`部署的时候，是吧该文件夹下的内容上传到了github的以用户名为仓库名的master分支上。
由于生成之后的文件是HTML格式的，不便于再编辑，所以存一份markdown格式的源代码也是很有必要的。在GitHub新建一个仓库hexoblog,使用这个仓库
存储Hexo的源代码。Coding.net与GitHub高度相似，不同的是它的Pages读取的是用户名为仓库名的gh-pages分支的代码，而不是master分支的代码，所以master分支正好可以用来
存储Hexo的源代码。

<!-- more -->

具体配置过程
![home](home.png)
`.git`中的`conf`文件中这样配置
``` bash
[core]
	repositoryformatversion = 0
	filemode = false
	bare = false
	logallrefupdates = true
	symlinks = false
	ignorecase = true
	hideDotFiles = dotGitOnly
[remote "origin"]
	url = https://username:password@git.coding.net/username/username.git
	url = https://github.com/username/hexoblog.git
	fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
	remote = origin
	merge = refs/heads/master
```
`.gitignore`文件中这样配置
``` bash
.DS_Store
Thumbs.db
db.json
*.log
node_modules/
public/
.deploy*/
```
`.deploy_git`文件夹中是执行`hexo d`提交的内容，在hexo配置文件`_config.yml`中这样配置
``` bash
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
 - type: git
   repo: https://username:password@git.coding.net/username/username.git
   branch: coding-pages
   message:
 - type: git
   repo: https://username:password@github.com/username/username.github.io.git
   branch: master
   message:
```
这样就可以实现生成的网站和网站源代码分别维护了。需要发布网站的时候，执行`hexo d`
需要备份博客数据的时候执行`git commit -a -m`和`git push origin master`。
>**注意**：把以上的username换成你自己的用户名，password换成自己的密码
