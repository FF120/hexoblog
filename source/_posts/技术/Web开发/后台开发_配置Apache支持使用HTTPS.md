---
title: 配置Apache支持使用HTTPS
toc: true

tags:
  - php
  - apache
date: 2016-06-11 20:20:13
---
## 1. 安装mod_ssl
``` bash
yum isntall mod_ssl
```

<!-- more -->

安装完成后再/etc/httpd/conf.d/下面有一个ssl.conf.打开查看下面两项配置的内容：
``` bash
SSLCertificateFile /etc/pki/tls/certs/localhost.crt
SSLCertificateKeyFile /etc/pki/tls/private/localhost.key
```
## 2. 生成密钥
``` bash
cd /etc/pki/tls/private
rm -f localhost.key
openssl genrsa 1024 > localhost.key
```
## 3. 生成证书
``` bash
cd /etc/pki/certs
rm -f localhhost.crt
openssl req -new -x509 -days -key ../private/localhost.key -out localhost.crt
```
然后填写证书的各项信息，证书生成。
重启Apache，就可以使用https访问了。
