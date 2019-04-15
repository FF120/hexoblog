---
title: PHP基本操作
toc: true

tags:
  - php
date: 2016-06-11 20:17:07
---
##1. 字符串的截取
取得`https://images.shiliujishi.com/sdlakfjadfosdfji.jpg`中的`sdlakfjadfosdfji.jpg`
<!-- more -->
```php
$url = strrchr ( $long_url, '/' ) ;
$url_length = -(strlen($url)-1);
$short_url = substr ($long_url, $url_length);
```
