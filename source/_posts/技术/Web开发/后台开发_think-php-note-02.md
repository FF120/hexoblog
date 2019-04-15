---
title: think php note 02
toc: true

tags:
  - thinkphp
date: 2016-06-11 16:00:12
---
# ThinkPHP入门教程-存入数据

## 配置显示输入表单的页面

``` php
testThinkPHP/Lib/Action/UserAction.class.php
public function addUserGet ()
{
    $this ->display();
}
```

<!-- more -->

## 新建模版 `testThinkPHP/Tpl/User/addUserGet.html`

``` html
<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">
    <title ></title>
</head>
<body>
    <form action="<?php echo U('User:addUserPost') ?>" method= "post">
        <input type= "text" name="user_name" />
        <input type= "password" name="password"/>
        <input type= "submit"/>
    </form >
</body>
</html>
```

在浏览器输入localhost/testThinkPHP/?m=User$a=addUserGet就可以看到表单输入页面了

## 单击submit按钮后，表单数据会提交到UserModel的addUserPost()方法

``` php
public function addUserPost ()
{
    $user_name = $_POST[ 'user_name'];
    $password = $_POST[ 'password'];
    $user = M('User');
    $data ['user_name'] = $user_name;
    $data ['password'] = $password;
    $user ->create($data );
    $user ->add();
}
```

执行完这个方法，数据库的user数据表中就会新增一条记录，但是要记住，id字段一定是自增的，否则可能会出现插入不进去的情况。
