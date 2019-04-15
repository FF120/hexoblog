---
title: 测试_使用Selenium测试UI
toc: true

tags:
  - UI测试
  - selenium
date: 2016-06-11 20:21:44
---
## 1. 安装Selenium Server
### 1.1 下载Server http://docs.seleniumhq.org/download/
选择Selenium Server（formerly the Selenium RC Server）下载。

<!-- more -->

### 1.2 安装Java环境，确保Java命令可以使用。
### 1.3 开启服务器
> `java -jar selenium-server-standalone-xxx.jar`

## 2. 安装 PHPUnit_Selenium Package
### 2.1 在项目的`composer.json`中的`require-dev`结点添加：`"phpunit/phpunit-selenium": ">=1.2"`

### 2.2 在项目目录运行 `composer update`

## 3. 编写测试用例检测环境是否安装正确
```php
    <?php
    class WebTest extends PHPUnit_Extensions_Selenium2TestCase
    {
        protected function setUp()
        {
            $this->setBrowser('firefox');
            $this->setBrowserUrl('http://www.example.com/');
        }

        public function testTitle()
        {
            $this->url('http://www.example.com/');
            $this->assertEquals('this is a title', $this->title());
        }

    }
    ?>
```

网页：
```
<html>
<head>
	<title>
		this is a title
	</title>
</head>
<body>
 this is the body.
</body>
```
返回
>Time: 2.56 seconds, Memory: 3.25Mb

>OK (1 test, 1 assertion)

环境配置成功。

## 4. 断言失败时保存网页截图
```php
<?php
require_once 'PHPUnit/Extensions/SeleniumTestCase.php';

class WebTest extends PHPUnit_Extensions_SeleniumTestCase
{
    protected $captureScreenshotOnFailure = TRUE;
    protected $screenshotPath = 'c:/xampp/htdocs/screenshots';
    protected $screenshotUrl = 'http://localhost/screenshots';

    protected function setUp()
    {
        $this->setBrowser('*firefox');
        $this->setBrowserUrl('http://www.example.com/');
    }

    public function testTitle()
    {
        $this->open('http://www.example.com/');
        $this->assertTitle('Example WWW Page');
    }
}
?>
```
你需要确保有screenshots文件夹，这样失败时就会保存网页截图到screenshots目录下。

## 5. 录制脚本测试UI
### 5.1 下载火狐插件Selenium IDE，安装
链接: `http://pan.baidu.com/s/1eQ1WkMI` 密码: `rt1d`
### 5.2 打开要测试的网页录制脚本
一个简单的测试页：
```html
<html>
<head>
	<title>
		this is a title
	</title>
</head>
<body>
  <form id = 'login'>
	<input id = 'user' type = 'text'/>
	<input id = 'pass' type = 'password'/>
	<input type = 'submit' value = 'login'/>
</body>
```
这里录制一个填写用户名密码的动作。然后在Selenium IDE中File->Export Test Case As->Java/Unit4/Remote Control导出，导出之后得到类似这样的代码：
```java
package com.example.tests;

import com.thoughtworks.selenium.*;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import java.util.regex.Pattern;

public class qq {
	private Selenium selenium;

	@Before
	public void setUp() throws Exception {
		selenium = new DefaultSelenium("localhost", 4444, "*chrome", "http://localhost/");
		selenium.start();
	}

	@Test
	public void testQq() throws Exception {
		selenium.open("/test/1.html");
		selenium.type("id=user", "123456");
		selenium.type("id=pass", "12");
		selenium.click("css=input[type=\"submit\"]");
		selenium.waitForPageToLoad("30000");
	}

	@After
	public void tearDown() throws Exception {
		selenium.stop();
	}
}
```
对照这里的代码，写完测试代码：
```php
<?php
require_once 'PHPUnit/Extensions/SeleniumTestCase.php';
class WebTest extends PHPUnit_Extensions_SeleniumTestCase
{
    protected $captureScreenshotOnFailure = TRUE;
    protected $screenshotPath = 'C:\xampp\htdocs\test\screenshots';
    protected $screenshotUrl = 'http://localhost/test/screenshots';

    protected function setUp()
    {
        $this->setBrowser('*firefox');
        $this->setBrowserUrl('http://localhost/');
    }

    public function testTitle()
    {
        $this->setSpeed('3000');
        $this->open('http://localhost/test/1.html');
        $this->type("id=user","123456");
        $this->type("id=pass","12");
        $this->click("css=input[type=\"submit\"]");
        $this->assertElementValueEquals("user", "123456");
        $this->assertElementValueEquals("pass", "12");
    }
}

?>
```
一般情况下只需要把setUp和Test中的代码对应到测试中即可，通常只需要改一下格式，函数名称都是通用的。
### 5.3 可用的断言
> - void assertElementValueEquals(string $locator, string $text)
> - void assertElementValueNotEquals(string $locator, string $text)
> - void assertElementValueContains(string $locator, string $text)
> - void assertElementValueNotContains(string $locator, string $text)
> - void assertElementContainsText(string $locator, string $text)
> - void assertElementNotContainsText(string $locator, string $text)
> - void assertSelectHasOption(string $selectLocator, string $option)
> - assertSelectNotHasOption(string $selectLocator, string $option)
> - assertTextPresent（检查在当前给用户显示的页面上是否有出现指定的文本）、
> - assertTextNotPresent（检查在当前给用户显示的页面上是否没有出现指定的文本）、
### 5.4 有用的设置
- 调整每个语句的执行速度 `$this->setSpeed('3000');`
- 使UI延长显示 `sleep(100);`
