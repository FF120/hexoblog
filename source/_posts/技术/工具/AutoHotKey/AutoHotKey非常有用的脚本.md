---
title: AutoHotKey非常有用的脚本
toc: true
tags:
  - AutoHotKey
date: 2018-12-02 17:42:38
---

键盘和鼠标按键修改工具AutoHotKey可以实现修改任意键盘按键和鼠标按键的功能，可以实现快捷短语的输入。

我的AutoHotKey脚本：

```
SetCapsLockState, AlwaysOff
;***********************************************************
;|+=======================================================+|
;||                 使用键盘模拟鼠标                      ||
;|+-------------------------+-----------------------------+|
;||      CapsLock+d         |    开启                     ||
;||      CapsLock+f         |    关闭                     ||
;|+-------------------------+-----------------------------+|
;||        d                |    左键                     ||
;||        f                |    右键                     ||
;||      ikjl               |    鼠标移动                 ||
;|+=======================================================+|
#SingleInstance
count = 0
JoyMultiplier = 0.20
JoyThreshold = 3
JoyThresholdUpper := 50 + JoyThresholdr
JoyThresholdLower := 50 - JoyThreshold
YAxisMultiplier = -1
SetTimer, WatchKeyboard, 20
Hotkey, d, ButtonRight
Hotkey, f, ButtonLeft
Hotkey, i,empty
Hotkey, k,empty
Hotkey, j,empty
Hotkey, l,empty
Hotkey, u,empty
Hotkey, m,empty
Return

CapsLock & d::
    SetTimer, WatchKeyboard,On
    Hotkey, d, on
    Hotkey, f, on
    Hotkey, i, on
    Hotkey, k, on
    Hotkey, j, on
    Hotkey, l, on
	Hotkey, u, on
	Hotkey, m, on
Return

CapsLock & f::
    SetTimer, WatchKeyboard, Off
    Hotkey, d, Off
    Hotkey, f, Off
    Hotkey, i, Off
    Hotkey, k, Off
    Hotkey, j, Off
    Hotkey, l, Off
	Hotkey, u, off
	Hotkey, m, off
Return

empty:
Return
WatchKeyboard:
MouseNeedsToBeMoved := false  ; Set default.
JoyMultiplier+=0.01
SetFormat, float, 03
i:=GetKeyState("i","p")
k:=GetKeyState("k","p")
j:=GetKeyState("j","p")
l:=GetKeyState("l","p")
u:=GetKeyState("u","p")
m:=GetKeyState("m","p")
if(u)
{
	send,{WheelUp}
}
if(m)
{
	send,{WheelDown}
}
if(l)
{
    MouseNeedsToBeMoved := true
    DeltaX := 10
}
else if(j)
{
    MouseNeedsToBeMoved := true
	BeMoved := true
    DeltaX := -10
}
else
    DeltaX = 0
if (i)
{
    MouseNeedsToBeMoved := true
    DeltaY := 10
}
else if (k)
{
    MouseNeedsToBeMoved := true
    DeltaY := -10
}
else
    DeltaY = 0
if MouseNeedsToBeMoved
{
    SetMouseDelay, -1  ; Makes movement smoother.
    MouseMove, DeltaX * JoyMultiplier, DeltaY * JoyMultiplier * YAxisMultiplier, 0, R
}
Else
count++
If(count>20){
JoyMultiplier = 0.30
count=0
}
return

ButtonLeft:
SetMouseDelay, -1  ; Makes movement smoother.
MouseClick, left,,, 1, 0, D  ; Hold down the left mouse button.
SetTimer, WaitForLeftButtonUp, 10
return

ButtonRight:
SetMouseDelay, -1  ; Makes movement smoother.
MouseClick, right,,, 1, 0, D  ; Hold down the right mouse button.
SetTimer, WaitForRightButtonUp, 10
return


WaitForLeftButtonUp:
if GetKeyState("f")
    return  ; The button is still, down, so keep waiting.
; Otherwise, the button has been released.
SetTimer, WaitForLeftButtonUp, off
SetMouseDelay, -1  ; Makes movement smoother.
MouseClick, left,,, 1, 0, U  ; Release the mouse button.
return

WaitForRightButtonUp:
if GetKeyState("d")
    return  ; The button is still, down, so keep waiting.
; Otherwise, the button has been released.
SetTimer, WaitForRightButtonUp, off
MouseClick, right,,, 1, 0, U  ; Release the mouse button.
return
;************************************************************
;|+=======================================================+|
;||                 使用键盘模拟鼠标                      ||
;|+-------------------------+-----------------------------+|
;||      a+f                |    左键                     ||
;||      a+d                |    右键                     ||
;||      a+u                |    向上滚轮                 ||
;||      a+m                |    向下滚轮                 ||
;||      a+i,j,k,l          |    鼠标移动                 ||
;|+=======================================================+|
;************************************************************
$a::
Send, a
return

a & i::
SetMouseDelay, -1 
MouseMove, 0, -15, 0, R 
return

a & k::
SetMouseDelay, -1 
MouseMove, 0, 15, 0, R
return

a & j::
SetMouseDelay, -1 
MouseMove, -15, 0, 0, R
return

a & l::
SetMouseDelay, -1
MouseMove 15 ,0,0,R
return

a & u::
Send, {WheelUp}
return

a & m::
Send, {WheelDown}
return

a & f::
SetMouseDelay, -1  
MouseClick, left,,, 1, 0, D  ; Hold down the left mouse button.
SetTimer, WaitForLeftButtonUp, 10
return

a & d::
SetMouseDelay, -1 
MouseClick, right,,, 1, 0, D  ; Hold down the right mouse button.
SetTimer, WaitForRightButtonUp, 10
return

;************************************************************
;|+=======================================================+|
;||                CapsLock改成Enter键                    ||
;|+-------------------------+-----------------------------+|
;||      CaspLock                |    Enter               ||
;||      alt + CapsLock          |    CaspLock            ||
;|+=======================================================+|
;************************************************************
$CapsLock::Enter 

LAlt & CapsLock::
GetKeyState, CapsLockState, CapsLock, T
if CapsLockState = D
	SetCapsLockState, AlwaysOff
else
	SetCapsLockState, AlwaysOn
KeyWait, CapsLock 
return

RAlt & CapsLock::
GetKeyState, CapsLockState, CapsLock, T
if CapsLockState = D
	SetCapsLockState, AlwaysOff
else
	SetCapsLockState, AlwaysOn
KeyWait, CapsLock 
return
;************************************************************
;|+=======================================================+|
;||                主键盘增加方向键                       ||
;|+-------------------------+-----------------------------+|
;||      CaspLock + i       |    Up                       ||
;||      CaspLock + k       |    Down                     ||
;||      CaspLock + j       |    Left                     ||
;||      CaspLock + l       |    Right                    ||
;||      CaspLock + u       |    Home                     ||
;||      CaspLock + m       |    End                      ||
;|+=======================================================+|
;************************************************************

CapsLock & k::
if GetKeyState("LShift", "P")
	Send, +{Down}
else if GetKeyState("LAlt", "P")
	Send,^{Down}
else if GetKeyState("LControl", "P")
	Send,^+{Down}
else
	Send, {Down}
Return

CapsLock & i::
if GetKeyState("LShift", "P")
	Send, +{Up}
else if GetKeyState("LAlt", "P")
	Send,^{Up}
else if GetKeyState("LControl", "P")
	Send,^+{Up}
else
	Send, {Up}
Return

; move left
CapsLock & j::
if GetKeyState("LShift", "P")
	Send, +{Left}
else if GetKeyState("LAlt", "P")
	Send,^{left}
else if GetKeyState("LControl", "P")
	Send,^+{left}
else
	Send, {Left}
Return

CapsLock & l::
if GetKeyState("LShift", "P")
	Send, +{Right}
else if GetKeyState("LAlt", "P")
	Send,^{Right}
else if GetKeyState("LControl", "P")
	Send,^+{Right}
else
	Send, {Right}
Return

CapsLock & u::
if GetKeyState("LShift", "P")
    Send, +{Home}
else if GetKeyState("LAlt", "P")
	Send, ^{Home}
else if GetKeyState("LControl", "P")
	Send, ^{Home}
else
	Send, {Home}
return

CapsLock & m::
if GetKeyState("LShift", "P")
    Send, +{End}
else if GetKeyState("LAlt", "P")
	Send, ^{End}
else if GetKeyState("LControl", "P")
	Send, ^{End}
else
	Send, {End}
return
;************************************************************
;|+=======================================================+|
;||                剪贴板设置成3个                        ||
;|+-------------------------+-----------------------------+|
;||      Ctrl + 1           |    复制到剪贴板1            ||
;||      Ctrl + 2           |    复制到剪贴板2            ||
;||      Ctrl + 3           |    复制到剪贴板3            ||
;||      Alt  + l           |    粘贴剪贴板1的内容        ||
;||      Alt  + 2           |    粘贴剪贴板2的内容        ||
;||      Alt  + 3           |    粘贴剪贴板3的内容        ||
;|+=======================================================+|
;************************************************************
^1:: 
Send ^c
a = %ClipBoard%
return

^2:: 
Send ^c
b = %ClipBoard%
return

^3:: 
Send ^c
c = %ClipBoard%
return

!1:: 
ClipBoard = %a%
Send ^v
return

!2:: 
ClipBoard = %b%
Send ^v
return

!3:: 
ClipBoard = %c%
Send ^v
return

;************************************************************
;|+=======================================================+|
;||                主键盘区域增加小键盘                   ||
;|+-------------------------+-----------------------------+|
;||      CapsLock + p       |    开关                     ||
;||      m                  |    0                        ||
;||      j, k, l            |    1, 2, 3                  ||
;||      u, i, p            |    4, 5, 6                  ||
;||      7, 8, 9            |    7, 8, 9                  ||
;|+=======================================================+|
;************************************************************
#if
	CapsLock & p::
	ONOFF := !ONOFF
Return
#if ONOFF
{
	u::4
	i::5
	o::6
	j::1
	k::2
	l::3
	m::0
	n::0
}
#if

;************************************************************
;|+=======================================================+|
;||                   快捷搜索                            ||
;|+-------------------------+-----------------------------+|
;||      CapsLock + e       |    翻译所选词汇             ||
;||      CapsLock + s       |    搜索所选词汇             ||
;|+=======================================================+|
;************************************************************

CapsLock & e::
ffSearchWord = %ClipBoard%
run https://fanyi.baidu.com/#en/zh/%ffSearchWord%
run https://cn.bing.com/dict/search?q=%ffSearchWord%
return

CapsLock & s::
Send ^c
content = %ClipBoard%
run https://www.baidu.com/s?wd=%content%
run https://cn.bing.com/search?q=%content%
return

CapsLock & n::
run http://www.baidu.com
return
;************************************************************
;|+=======================================================+|
;||                   重新加载本脚本                      ||
;|+-------------------------+-----------------------------+|
;||      CapsLock + r       |    重载                     ||
;|+=======================================================+|
;************************************************************
CapsLock & r::
Send ^s
reload
return
```

