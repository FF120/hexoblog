---
title: Hexo的使用04-数学公式
toc: true

tags:
  - latex
  - hexo
date: 2017-04-12 16:36:21
---

数学公式是我们经常需要使用到的表达工具，只是用键盘上的符号有时候很难表达清楚想要表达的公式，格式也不是很友好。Hexo可以支持Latex公式的显示，我们只需要经过简单的配置就可以使Hexo支持Latex公式的显示。当然，通过使用atom的插件，Latex公式的实时预览也很容易实现，这样本地写作环境和提交之后的格式就是完全一致的。

<!-- more-->
## 安装可以实时预览公式的atom插件

https://atom.io/packages/markdown-preview-enhanced

## 安装hexo-math

https://github.com/akfish/hexo-math

## 常用的表示

### 希腊字母

|名称|Tex书写形式|效果|
|--|--|--|
|alpha|\alpha|$\alpha$|
|beta|\beta|$\beta$|
|gamma|\gamma|$\gamma$|
|theta|\theta|$\theta$|
|mu|\mu|$\mu$|
|pi|\pi|$\pi$|
|rho|\rho|$\rho$|
|sigma|\sigma|$\sigma$|
|phi|\phi|$\phi$|
|omega|\omega|$\omega$|
|chi|\chi|$\chi$|
|psi|\psi|$\psi$|

### 上标和下标

上标用`^`表示，下标用`_`表示，需要注意的是，这里的上标和下标的作用范围是紧紧跟在后面的字母或者数字，如果需要多个，需要用`{}`括起来。例如：`x^56`会得到$x^56$,想要得到正确的结果，需要这样书写，`x^{56}`,这样得到的结果就是$x^{56}$. 在不用括号会引起歧义的地方必须使用括号，否则编译无法通过，例如`x^5^6`.

### 括号

小括号和中括号没有用来表示特殊的含义，所以可以直接使用。大括号的使用需要使用转义字符。

### 数学运算

|书写格式|显示效果|
|--|--|
|\arcsin|$\arcsin(xy)$|
|\sin|$\sin(x^5)$|
|\arccos|$\arccos(x-y)$|
|\cos|$\cos(xyz)$|
|\arctan|$\arctan(xy)$|
|\arg|$\arg()$|
|\cosh|$\cosh$|
|\sinh|$sinh$|
|\tanh|$tanh$|
|\int|$\int$|
|\iint|$\iint$|
|\iiint|$\iiint$|
|\oint|$\oint$|
|\coprod|$\coprod$|
|\bigvee|$\bigvee$|
|\bigwedge|$\bigwedge$|
|\biguplus|$\biguplus$|
|\bigca|$\bigcap$|
|\bigcup|$\bigcup$|
|\intop|$\intop$|
|\prod|$\prod$|
|\sum|$\sum$|
|\bigoplus|$\bigoplus$|
|\smallint|$\smallint$|
|\bigodot|$\bigodot$|
|\odot|$\odot$|
|\bigotimes|$\bigotimes$|
|\bigsqcup|$\bigsqcup$|

### 逻辑运算符

|书写格式|显示效果
|--|--|
|\forall|$\forall$|
|\exists|$\exists$|
|\nexists|$\nexists$|
|\therefore|$\therefore$|
|\because|$\because$|

### 分数

|书写格式|显示效果|示例|
|--|--|--|
|\{10}frac{56}{78}|${10}\frac{56}{78}$|
|4\dfrac89|$4\dfrac89$|
|54\over34|$54\over34$|
|10\tfrac89|$10\tfrac89$|

### 大括号

```
$$
f(x)=\begin{cases}
        0   &  \text{x>0}\\
        1   &  \text{x<=0}
      \end{cases}
$$
```

$$
f(x)=\begin{cases}
        0   &  \text{x>0}\\
        1   &  \text{x<=0}
      \end{cases}
$$

公式详细的简介：
http://mlworks.cn/posts/introduction-to-mathjax-and-latex-expression/

Katex支持的所有操作符：
https://github.com/Khan/KaTeX/wiki/Function-Support-in-KaTeX

latex手写符号识别系统：
http://detexify.kirelabs.org/classify.html
