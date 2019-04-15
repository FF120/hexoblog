---
title: Make phpStorm friendly to laravel
date: 2016-06-11 15:15:12
toc: true

tags:
- phpStorm
- laravel
---
# phpStorm的个性化设置
1. `file-->setting` 打开设置界面，有两个`appearance`,第一个设置软件整体的风格，例如可以改成Windows风格

<!-- more -->


![phpStorm setting show](setting_show.png)
![phpStorm windows style](phpStorm_windows_style.png)
2. 第二个`appearance`是修改编辑器风格的，我喜欢修改成sublime Text的风格
- 下载主题包：链接: http://pan.baidu.com/s/1nthy0kT 密码: 2ymp
- 将里面的XML结尾的文件复制到`C:\Users\Seemeloo1\.WebIde70\config\colors` 文件夹下，这里的Seemeloo1是用户名，不同的电脑不一样。
- 在主题选择页面找到刚才下载的主题，另存为，然后修改字体大小和其它属性

# 安装laravel-ide-helper实现代码自动完成、代码提示和跟踪
1. 在`composer.json`的`require`下添加一行
```
“barryvdh/laravel-ide-helper”:”1.11.*”
```
2. 执行`composer update`安装刚才添加的插件
3. 执行`php artisan ide-helper:generate`,生成代码提示和跟踪需要的文件，现在就可以按住Ctrl单击方法追踪方法来源了。
![ide-help](ide_helper.png)
