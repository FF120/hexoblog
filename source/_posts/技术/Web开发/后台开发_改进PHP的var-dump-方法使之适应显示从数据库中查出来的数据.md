---
title: 改进PHP的var_dump()方法使之适应显示从数据库中查出来的数据
toc: true

tags:
  - PHP
date: 2016-06-23 10:44:06
---
改进PHP的var_dump使之拥有良好的显示格式。
<!-- more -->
```php
/**
 * array(7) {
[0] => array(32) {
["order_id"] => string(2) "34"
["shipping_time"] => string(1) "0"
["created_at"] => string(10) "1433565988"
}
[1] => array(32) {
["order_id"] => string(2) "35"
["shipping_time"] => string(1) "0"
["created_at"] => string(10) "1433573933"
}
[2] => array(32) {
["order_id"] => string(2) "36"
["order_sn"] => string(9) "BJA000036"
["wx_trade_no"] => string(32) "wx712cd30713b968c114336440083628"
}
 * 形如上面这样的数组，使用var_dump()函数打印的时候会打印出所有的数据项，尤其是在数据库字段较多的情况下，
 * 不便于找到自己感兴趣的数据。
 * 本方法改进了var_dump()方法，可以指定要打印的键。例如想看从order表中查处的order_id和city_id,
 * 可以这样使用：du($order,['order_id','city_id'])
 * 输出如下格式：
 *array[7] {
[0] =>array[32] {
[order_id] => string(2) : 34
[city_id] => string(1) : 0
}
[1] =>array[32] {
[order_id] => string(2) : 35
[city_id] => string(1) : 0
}
[2] =>array[32] {
[order_id] => string(2) : 36
[city_id] => string(1) : 0
}
[3] =>array[32] {
[order_id] => string(2) : 37
[city_id] => string(1) : 0
}
}
 * @param $data 要显示结构的变量
 * @param $keys 要显示的键，为空则显示所有的键
 * @param $echo 是否输出结果到浏览器
 */
function du($data,$keys='',$echo=true)
{
    $message = "";
    $message .= gettype($data) . "[" . count($data) . "] {" . "<br>";
    foreach ($data as $cc => $c) {
        $message .= " [" . $cc . "] =>" . gettype($c) . "[" . count($c) . "] {" . "<br>";
        foreach ($c as $key => $value) {
            if (empty($keys)) {
                $message .= "  [" . $key . "]" . " => " . gettype($value) . "(" . strlen($value) . ")" . " : ".$value;
                $message .= "<br>";
            } else if (in_array($key, $keys)) {
                $message .= "  [" . $key . "]" . " => " . gettype($value) . "(" . strlen($value) . ")" . " : ".$value;
                $message .= "<br>";
            }
        }
        $message .= "  }";
        $message .= "<br>";
    }
    $message .= "}";
    if ($echo) {
        echo $message;
    } else {
        $a = str_replace("<br>","\r\n",$message);
        $a = str_replace(" ","  ",$a);
        return $a;
    }
}
```
