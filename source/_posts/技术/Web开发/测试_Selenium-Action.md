---
title: 测试_Selenium Action
toc: true

tags:
  - selenium
date: 2016-06-11 20:36:31
---
action用于模拟用户与浏览器之间的交互
<!-- more -->
> - open(url) 打开网页
> - click() 单击
> - clickAndWait() 单击并等待，生成代码时，比上一句多一个`$this-waitForPageLoad('3000');`
> - type( , ) 输入文本
> - select() 选择下拉菜单
> - goBack() 单击浏览器返回按钮
> - close() 模拟单击关闭按钮
> - setSpeed() 设置执行速度
> - pause() 等待指定的时间后继续执行
> - setTinmeout()  指定过期时间
