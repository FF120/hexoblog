---
title: 测试_Selenium定位元素的几种方式
toc: true

tags:
  - selenium
date: 2016-06-11 20:34:48
---

## 1. 默认方式
形如`identifier = location`的定位方式在HTNL文件中寻找第一个ID等于location的元素，如果没有，则匹配第一个name等于location的元素，也可以直接简写成`location`

<!-- more -->

## 2. 通过ID定位
形如`id = id_name`的定位方式在HTML文件中匹配ID等于id_name的元素。
## 3. 通过name定位
形如`name = name_name`的定位方式可以匹配文件中第一个name等于name_name的元素。通常，在HTNL中，name与ID不同，name并不一定是唯一的，可能许多不同的元素有相同的name，为了匹配到不同的元素，可以加上不同于其他元素的特征，例如：
```
<html>
  <body>
   <form id="loginForm">
    <input name="username" type="text" />
    <input name="password" type="password" />
    <input name="continue" type="submit" value="Login" />
    <input name="continue" type="button" value="Clear" />
   </form>
 </body>
 <html>
 ```
 -----------------
 > - name = username 定位到第四行
 > - name = continue value = clear
 name = continue clear
 name = continue type=button
 都会定位到第七行。

通过上面的三种方法，可以独立的测试某个元素，与HTML结构没有关系。所以，当你的网页结构经常变化又希望不要经常改变测试代码的时候，尽量使用上面三种方式就显得十分重要。
## 4. 通过XPath定位
- XPath是XML中定位结点的一种方式，因为HTML也可以实现XML接口，所以也可以XPath语法定位HTML元素。
- XPath有两种定位方式，绝对定位和相对定位。
  - 绝对定位从`html`元素开始一级一级找到需要的元素，HTML文件很小的改动也可能打破这种层级关系，所以不推荐使用这种定位方式。
  - 相对定位是从某一个容易定位的元素开始，以它为参照找到需要的元素，原HTML文件部分改动时，对这种相对关系影响较小，推荐使用这种定位方式。
- 只有当用以上三种方式不容易定位到需要的元素时，才推荐使用XPath定位。
使用上面的HTML举几个例子：
> 1. xpath = /html/body/form[1]
绝对定位，定位到第三行form
> 2. //form[1]
相对定位，定位到HTML中第一个form元素
> 3. xpath=//form[@id='loginForm']
定位到`id='loginForm'`的form元素
> 4. xpath=//form[input/@name='username']
定位到有一个`name=username`的input元素的form中
> 5. //input[@name='username']
第一个name=username的input元素
> 6. //form[@id='loginForm']/input[1]
定位到Id='loginForm'的form中的第一个input元素
> 7. //input[@name='continue'][@type='button']
定位到name=continue type=button的input元素
> 8. //form[@id='loginForm']/input[4]
定位到ID=loginForm的form元素中的第四个元素。

## 5. 通过超链接定位
适合定位带超链接的元素
> 1. link=www.baidu.com
> 2. link=首页
如果HTML中存在多个href相同的字段，总是返回第一个。
## 6. 通过DOM定位
使用JavaScript定位元素的方式：
>dom=document.getElementById('loginForm') (3)
    dom=document.forms['loginForm'] (3)
    dom=document.forms[0] (3)
    document.forms[0].username (4)
    document.forms[0].elements['username'] (4)
    document.forms[0].elements[0] (4)
    document.forms[0].elements[3] (7)
## 7. 通过CSS样式定位
CSS通过选择器将定义的样式与文档中的元素绑定在一起。CSS定位元素的策略也可以在Selenium中使用。

    css=form#loginForm (3)
    css=input[name="username"] (4)
    css=input.required[type="text"] (4)
    css=input.passfield (5)
    css=#loginForm input[type="button"] (7)
    css=#loginForm input:nth-child(2) (5)


----------

小括号中的数字代码定位到HTML代码的第几行。
