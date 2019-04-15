---
title: think php note 01
toc: true

tags:
  - thinkphp
date: 2016-06-11 15:57:25
---
# ThinkPHP入门教程-默认的规则
## URL模式
在testThinkPHP/Conf/config.php中加入
'URL_MODEL' => 0,  //URL模式
此时的URL模式是普通模式
http://serverName/appName/?m=module&a=action&id=1
serverName： localhost
appName: testThinkPHP 就是项目文件夹的名字
m = 控制器的名字，例如UserAction.class.php的名字为User
a = 控制器方法的名字，想执行showUser方法，就写showUser
后面的参数为传入该方法的参数
所以采用此种方式 访问刚才显示用户名密码的页面的 地址就是：
http://localhost/testThinkPHP/?m=User&a=showUser

<!-- more -->

'URL_MODEL' => 1,  //URL模式
这个时候是PATHINFO模式
http://serverName/appName/module/action/id/1/
这个时候访问显示用户名密码的页面的地址就是：
http://localhost/a/index.php/User/showUser

值得注意的是，设置URL_MODEL只是改变使用系统U函数生成的URL的样式，无论设置成什么模式，几种URL都是可以访问的。
例如，当URL_MODEL = 1 的时候，访问
http://localhost/testThinkPHP/?m=User&a=showUser
依然可以出现显示用户名密码的界面

## 模型-控制器-视图默认的映射关系

testThinkPHP
     --Lib
          --Action
               --UserAction.class.php
          --Model
               --UserModel.class.php
     --Tpl
          --User
               --addUserGet.html
               --addUserPost.html
               --showUser.html
testThinkPHP是项目的主目录，按照这样的文件夹结构，定义了模型User，控制器User，视图组User
控制器User中的方法名称自动对应视图组User中的同方法名称相同的模版文件。

在这样的默认对应规则下，可以使用User:addUserGet找到模板文件testThinkPHP/Tpl/User/addUserGet.html文件

## 配置文件的生效位置

testThinkPHP/Conf/config.php 中返回数组，配置参数就会生效。
