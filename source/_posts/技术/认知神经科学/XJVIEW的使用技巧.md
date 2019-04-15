---
title: XJVIEW的使用技巧
toc: true

tags:
  - XJVIEW
date: 2017-04-14 09:53:17
---
xjview是用来查看大脑激活图像的比较好用的软件，它基于matlab和SPM开发而成。如果需要批量处理一些数据，可能设计到提取里面的代码。下面介绍一些xjview的命令用法。
<!-- more -->
### 显示激活图像
```c
spmT_filepath = 'D:\spmF_0001.nii';
xjview(spmT_filepath); %显示一幅激活图像，显示多幅可以用逗号隔开
```

### 找到激活最大值
```c
h = spm_mip_ui('FindMIPax');
% loc     - String defining jump: 'dntmv' - don't move
%                                 'nrvox' - nearest suprathreshold voxel
%                                 'nrmax' - nearest local maxima
%                                 'glmax' - global maxima
loc = 'glmax';
xyz = spm_mip_ui('Jump',h,loc); % 更换loc参数，实现不同的找到最大值的方法。
```

### 根据某个坐标找到局部激活最大值
```c
h = spm_mip_ui('FindMIPax');
hC = 0;
[xyz,d] = spm_mip_ui('SetCoords',[-14,-26,-6],h,hC);
% xyz是找出来的坐标，d是移动的距离
```

### 查看某个坐标在图像中的位置
```c
xjview([20 10 1],[10]); % 后面的[10]是赋值给该点的激活强度，为了颜色的显示。
```
