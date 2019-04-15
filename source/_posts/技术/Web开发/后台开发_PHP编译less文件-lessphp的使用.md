---
title: PHP编译less文件-lessphp的使用
toc: true

tags:
  - php
  - less
date: 2016-06-11 19:56:51
---
## 用composer安装
``` bash
{
    "require":{
        "leafo/lessphp": "0.4.0"
    }
}
```

<!-- more -->

手动安装
[lessphp源代码](https://github.com/leafo/lessphp)
下载源代码，将其中的lessc.inc.php拷贝到任意文件夹中，使用时include即可。
[lessphp文档地址](http://leafo.net/lessphp/docs/)
##2. 使用
根据变量的不同将style.less编译成不同的css文件
``` php
require 'app\lib\lessc.inc.php';
use lessc;

 $less = new lessc();
        $less->setVariables(array(
            "color-main" => "#555555",
            "color-secondary" => "#222222",
            "color-contract" => "#333333"
        ));
        $inputFile = "public_html/styles/style.less";
        $outputFile = "public_html/styles/style.css";
        $inputString = file_get_contents($inputFile);
        $ouputString =  $less->compile($inputString);
        echo file_put_contents($outputFile,$ouputString);
```
