---
title: Linux主机之间同步文件
toc: true

tags:
  - linux
date: 2016-06-11 20:10:39
---
## 1. scp命令
在主机A上操作
### 1.1 主机A复制文件到主机B
> - scp `/home/filename` `root@101.1.1.3:/home`

###1.2 从主机B复制文件到主机A
> - scp `/root/fei/a.txt` `root@121.42.43.161:/root/fei`

<!-- more -->

###1.3 避免每次都需要输入密码
####1.3.1 生成密钥
> ssh-keygen -t rsa
查看当前用户目录下.ssh文件夹
> -rw------- 1 root root 1671 Sep 28 14:34 id_rsa
> -rw------- 1 root root  400 Sep 28 14:34 id_rsa.pub

####1.3.2 上传密钥到服务器B
> scp `/root/.ssh/id_rsa.pub` `10.0.0.2:/root/.ssh/authorized_keys`
将本地生成的公钥上传到B，并改名authorized_keys。
4. 写成脚本执行，传送批量文件
> 1. touch upload.sh
> 2. vi upload.sh
>\#!/bash/sh
scp `/root/fei/a.txt` `10.1.1.3:/root/fei/`
> 3. chmod u+x upload.sh
> 4. ./upload.sh

### 1.4 上传文件夹
> scp -r  添加-r参数即可

## 2. rsync 命令
###2.1 安装rsync服务
一般都已经安装rsync服务，只需要自己添加配置文件即可。ubuntu 安装方式：
`sudo apt-get install rsync`
###2.2 配置文件（主机A）
- touch /etc/rsyncd.conf
创建服务器配置文件
```sh
pid file = /var/run/rsyncd.pid
port = 873
address = 114.215.128.207 (服务器的IP，就是A主机的IP)
uid = root
gid = root
use chroot = yes
read only = yes
\#hosts allow=192.168.1.0/255.255.255.0
\#hosts deny = *
max connection = 5
motd file = /etc/rsyncd.motd
log file = /var/log/rsyncd.log
transfer logging = yes
log format = %t %a %m %f %b
syslog facility = local3
timeout = 300

[test]（备份结点）
path = /root/fei（需要备份的路径）
list=yes
ignore errors
auth users = root
secrets file = /etc/rsyncd.secrets
（客户端使用rsync命令时，需要使用这里设定的用户名和密码）
comment = This is test data（备份的注释）
exclude = （忽略的文件夹）
```
- touch /etc/rsyncd.secrets
创建密码文件
```
root:123456(用户名：密码)
```
- touch /etc/rsyncd.motd
创建服务器信息文件
```
++++++++++++++++
welcome to use rsync service !
(自定义当客户端连接时返回的提示信息)
+++++++++++++++++
```
- 修改权限
chown root.root rsyncd.secrets
chmod 600 rsyncd.secrets
### 2.3 启动服务
```
rsync --daemon --config = /etc/rsyncd.cond
(如果配置文件就在etc下，可以省略--config)
```
### 2.4 在主机B上访问rsync服务
#### 2.4.1 查看主机A上提供的备份数据源
```
rsync --list-only root@123.234.321.1::
```
出现配置文件中定义的欢迎信息说明配置成功。
```
+++++++++++++++++++
Welcome to use rsync service !
++++++++++++++++++++++++

test            This is test data
test就是主机A或者说rsync服务器上提供的数据源。如果在数据源的配置文件中加上`list=no`则不会显示在这里
```
#### 2.4.2 查看数据源的内容
-`rsync -avzP root@114.215.2.207::test`这里会要求输入密码，密码就是在主机A中的rsyncd.secrets文件中保存的密码。
####2.4.3 把A中的内容同步到B
```
 rsync -avzP root@114.215.128.207::test test
 (同步到本机当前目录的test文件夹下)
 rsync -avzP --delete root@114.215.128.207::test test
 (加上delete参数A中删除的文件也会在B中删除)
```
#### 2.4.4 避免每次同步都需要输入密码
在本地创建密码文件
```
touch rsyncd.secrets;chmod 600 rsyncd.secrets;echo 'password'>rsyncd.secrets
同步时运行
rsync -avzP --delete --password-file=rsyncd.secrets root@114.215.128.207::test test
```
#### 2.4.5 编写脚本实现同步ls
```
#!/bin/sh
/usr/bin/rsync -avzP --password-file=/root/fei/rsyncd.secrets root@114.215.128.207::test
```
#### 2.4.6 定时执行同步脚本实现自动同步
```
crontab -e (打开定时任务编辑器)
输入一下内容：
`* * * * *　／root/fei/rsync.sh`
(每分钟自动运行rsync.sh)
```
> crontab基本操作
crontab -l 列出当前的定时任务
crontab -r 删除当前的定时任务
crontab -e 编辑当前的定时任务
- 定时的格式：
f1 f2 f3 f4 f5 program
其中 f1 是表示分钟，f2 表示小时，f3 表示一个月份中的第几日，f4 表示月份，f5 表示一个星期中的第几天。program 表示要执行的程序。
当 f1 为 * 时表示每分钟都要执行 program，f2 为 * 时表示每小时都要执行程序，其馀类推
当 f1 为 a-b 时表示从第 a 分钟到第 b 分钟这段时间内要执行，f2 为 a-b 时表示从第 a 到第 b 小时都要执行，其馀类推
当 f1 为 */n 时表示每 n 分钟个时间间隔执行一次，f2 为 */n 表示每 n 小时个时间间隔执行一次，其馀类推
当 f1 为 a, b, c,... 时表示第 a, b, c,... 分钟要执行，f2 为 a, b, c,... 时表示第 a, b, c...个小时要执行，其馀类推
使用者也可以将所有的设定先存放在档案 file 中，用 crontab file 的方式来设定时程表。
当程式在你所指定的时间执行后,系统会寄一封信给你,显示该程式执行的内容,若是你不希望收到这样的信,请在每一行空一格之后加上   >   /dev/null   2>&1   即可。

#### 2.4.7 只复制目录结构，不复制内容
```
rsync -av --include '*/' --exclude '*' source-dir dest-dir
```
