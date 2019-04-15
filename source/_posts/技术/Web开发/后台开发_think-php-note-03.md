---
title: think php note 03
toc: true

tags:
  - thinkphp
date: 2016-06-11 16:02:38
---
# ThinkPHP入门教程-取出数据

## 配置数据库连接
- 本地mysql数据库的配置如下：
   数据库用户名：root
   密码：空
   使用的数据库：test
   数据库的信息：只有一张表user,user包含id,name,password三个字段
- 在testThinkPHP/Conf/config.php中键入如下代码：


<!-- more -->

``` php
<?php
return array (
          //'配置项'=>'配置值'
    'DB_TYPE'   => 'mysql', // 数据库类型
    'DB_PORT'   => '3306', // 端口
    'DB_CHARSET' => 'utf8', // 数据库编码默认采用utf8
    'DB_HOST'   => "localhost", // 服务器地址
    'DB_NAME'   => "test", // 数据库名
    'DB_USER'   => "root", // 用户名
    'DB_PWD'    => "123456",  // 密码
    'DB_PREFIX' => "a_", // 数据库表前缀
    );
?>
```

## 定义模型

在testThinkPHP/Lib/Model/UserMdel.php 中键入如下内容：

``` php
<?php
/**
* Created by PhpStorm.
* User: Administrator
* Date: 2015/7/18
* Time: 17:36
*/
class UserModel extends Model {

}
```

以上表示定义了User模型，该模型会自动和数据库中的user数据表相对应

## 定义控制器

在ThinkPHP/Lib/Action/UserAction.class.php 中：

``` php
<?php

class UserAction extends Action {

    public function showUser()
    {
        $user = M('User');
        $user ->find(1 );
        $user_name = $user->user_name ;
        $password = $user->password ;
        $tpl = "User:showUser";
        $this ->assign('username', $user_name);
        $this ->assign('password', $password);
        $this ->display($tpl );
    }
}
```
## 定义视图

ThinkPHP/Tpl/User/showUser.html 中键入：

``` html
<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">
    <title ></title>
</head>
<body>
    用户名： <?php echo $username?>  < br>
    密码： <?php echo $password?> < br>
</body>
</html>
```

## 查看结果

在浏览器中输入 localhost/testThinkPHP/?m=User&a=showUser

至此，我们完成了一个简单的从数据库中读取数据，显示在视图中的过程，下面再完成一个从视图中读取数据，存入数据库的过程，框架的基本功能就算是掌握了。
