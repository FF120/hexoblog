---
title: Redis入门
toc: true

tags:
  - redis
date: 2016-06-11 20:33:14
---
##1. 安装
- Windows 平台
下载地址：链接: http://pan.baidu.com/s/1ntkhsxF 密码: 9c27

<!-- more -->
32位系统选择redisbin_x32.zip,64位系统选择redisbin_x64.zip,解压到任意文件夹，文件结构如下：
> redis-server.exe   服务器端程序
> redis-cli.exe 客户端程序，用来连接服务器
> redis-check-dump.exe 本地数据库检查程序
> resdis-benchmark.exe 性能测试工具

##2. 启动服务
`redis-server.exe redis.conf`
##3. 连接服务器
`redis-cli.exe -h 127.0.0.1 -p 6379`
##4. 基本操作
- set key value 插入数据
- get key 获取数据
- mget key1 key2 key3 一次获取多个数据
- del key 删除数据
- exits key 判断是否存在
- select 0 选择第一个数据库（默认有0-15共16个数据库）
- keys * 查看所有的key
