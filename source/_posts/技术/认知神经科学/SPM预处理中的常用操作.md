---
title: SPM预处理中的常用操作
toc: true

tags:
  - fMRI
date: 2017-03-20 20:59:07
---
一些数据处理当中用的到的功能函数。
<!-- more -->

## 比较多幅图像是否配准
```matlab
function check_img(imgs)
% -------------------------------------------------------------------------
% 功能：比较多幅图像
% 调用：check_img(imgs)
% 参数：
%   imgs: cell类型的图像
% 示例：
%   c1_img = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160713001\t1_mprage_sag_p2_0026\run1\c1s20160713001-193508-00001-00176-1.img';
%   c2_img = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160713001\t1_mprage_sag_p2_0026\run1\c2s20160713001-193508-00001-00176-1.img';
%   c3_img = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160713001\t1_mprage_sag_p2_0026\run1\c3s20160713001-193508-00001-00176-1.img';
%   func_img = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160713001\ep2d_bold_moco_p2_rest_0006\f20160713001-190315-00238-00238-1.img';
%   imgs = {c1_img,c2_img,c3_img,func_img};
%   check_img(imgs);
% -------------------------------------------------------------------------
spm_jobman('initcfg')
matlabbatch{1}.spm.util.checkreg.data = imgs;
spm_jobman('run',matlabbatch);
```

## 批量复制文件夹结构
```matlab 
function copy_structure(origin_path,destination_path,fileFilter,varargin)
% -------------------------------------------------------------------------
% 功能： 复制目录结构 或者 复制文件
% 调用1：copy_structure(origin_path,destination_path,filter)
% 调用2：copy_structure(origin_path,destination_path,fileFilter,subExpIDFilter)
% 调用3：copy_structure(origin_path,destination_path,fileFilter,subExpIDFilter，runExpIDFilter)
% 参数：
%   origin_path : 要复制的目录结构的绝对路径
%   destination_path : 新文件夹绝对路径
%   subExpIDFilter ：子一级目录通配符
%   runExpIDFilter : 子二级目录通配符
%   fileFilter: 过滤器，决定拷贝哪些文件
% 示例：
%   copy_structure(origin_path,destination_path,'s4w*')
%   copy_structure(origin_path,destination_path,'s4w*','20160916001*')
%   copy_structure(origin_path,destination_path,'s4w*','20160916001*','ep2d_bold_moco_p2_rest_0006*')
% 说明： subExpIDFilter默认值为'2016*';runExpIDFilter 默认值为 'ep2d*'
% --------------------------------------------------------------------------
if nargin < 4
   subExpIDFilter = '2016*';% 被试文件夹通配符 
else
    subExpIDFilter = varargin{1};
end
if nargin < 5
    runExpIDFilter = 'ep2d*';% RUN文件夹通配符
else
    runExpIDFilter = varargin{2};
end
%fileFilter = 'w*';
cd(origin_path);
subExpID=dir(subExpIDFilter); 
for i=1:size(subExpID,1)
    mkdir([destination_path,subExpID(i).name]);
    cd([origin_path,subExpID(i).name]);
    runExpID=dir(runExpIDFilter); 
    for j=1:size(runExpID,1)
        mkdir([destination_path,subExpID(i).name,'\',runExpID(j).name]);
        cd([origin_path,subExpID(i).name,'\',runExpID(j).name]);
        if nargin > 2
            copyfile(fileFilter,[destination_path,subExpID(i).name,'\',runExpID(j).name],'f');
        end
    end
end
```

### 从多标签mask生成单标签mask
注意本函数依赖marsbar, 需要将marsbar工具包setpath之后才能使用。
```matlab
function create_masks_from_multiple_labels_img()
% -------------------------------------------------------------------------
% 功能： 从多标签图像创建多个Mask
% 参数：
%   multiple_label_img_path ：多标签图像路径
%   multiple_label_path : 标签路径，名称为ROI，包含ID，Nom_C,Nom_L,ID是标签，整数，剩下的是名称
%   save_path ： 生成Mask的保存路径
%   P ： 提供重新采样的参数，需要随便一张被试的图像，Mask会按照该图像的规格重新采样

aal = 'C:\mazcx\matlabtoolbox\spm8\toolbox\wfu_pickatlas\MNI_atlas_templates\TD_label_extended_modified.img';
aal_label = 'C:\mazcx\matlabtoolbox\spm8\toolbox\wfu_pickatlas\MNI_atlas_templates\TD_label_extended_modified_List.mat';


multiple_label_img_path = aal;
multiple_label_path = aal_label;
save_path = 'D:\FMRI_ROOT\YANTAI\DESIGN\MASK\TDLabels\';

P = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\wraf20160911002-182754-00008-00008-1.img';
roi_space = spm_vol(P);
% ROI names
load(multiple_label_path);
marsbar('on');
% Make ROIs
vol = spm_vol(multiple_label_img_path);
for r = 1:length(ROI)
  nom = ROI(r).Nom_L;
  func = sprintf('img == %d', ROI(r).ID); 
  o = maroi_image(struct('vol', vol, 'binarize',1,...
			 'func', func, 'descrip', nom, ...
			 'label', nom));
  cd(save_path);
  %saveroi(maroi_matrix(o), fullfile(roi_path,['MNI_' nom '_roi.mat']));
  mars_rois2img(maroi_matrix(o),['MNI_' nom '.img'],roi_space);
end
```

## mask乘以激活之后再做成mask,就是每个被试不同的mask
```matlab
function create_mask_use_T(input_img,output_img,f)
% 制作的mask乘以相应的激活之后再做成Mask
% STG_mask_path = 'D:\FMRI_ROOT\YANTAI\DESIGN\MASK\NiftiPairs_Resliced_STG.mn.img';
% spmT_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level\20160911002\spmT_0017.img';
% input_img = {STG_mask_path; spmT_path}; % 一个mask , 一个spmT图像
% output_img = 'd:\aaa.img';
%------------
P = input_img;
Q = output_img;
%f = 'i1.*(i2>3.0987)';
dmtx = 0;
mask = 0;
type = 4;
hold = 0;
flags = {dmtx,mask,type,hold};
[Q,Vo] = spm_imcalc_extend(P,Q,f,flags);
```
## 将同一个被试不同RUN的头动文件合并在一起
```matlab
% 将多个rp*头动文件，合成一个头动文件
pre_processing = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\';
cd(pre_processing);
subjects = dir('2016*');
for i = 1: size(subjects,1)
   cd([pre_processing,subjects(i).name]);
   SubRunID = dir('ep2d*');
   rps = ones(272*4,6);
   for j = 1:size(SubRunID,1)
       cd([pre_processing,subjects(i).name,'\',SubRunID(j).name]);
       file = dir('rp*');
       filename = file(1).name;
       a = load(filename);
       rps(272*(j-1)+1:272*j,1:6) = a;
   end
    cd([pre_processing,subjects(i).name]);
    save('rp_all.txt','rps','-ascii');
end
```

## 提取图像某个点的值并绘制时间序列曲线
```matlab
function Y = extract_time_series(V,XYZ)
% -------------------------------------------------------------------------
% 功能：提取图像某个点的值并绘制时间序列曲线
% 调用：Y = plot_time_series(V,XYZ)
% 参数： 
%   XYZ ：三维坐标，图像中的点
%   V : 存放图像路径的cell
% 示例：
%   XYZ = [13;48;2];
%   V = {'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\af20160911002-182750-00006-00006-1.img',
%    'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\af20160911002-182750-00006-00006-1.img'};
%   Y = plot_time_series(V,XYZ)
% -------------------------------------------------------------------------
Y = spm_get_data(V,XYZ);
```

## 使用xjview批量找到激活的峰值并保存
```matlab

spmT_filepath = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level\20160911002\spmT_0020.hdr';
xjview(spmT_filepath);
%显示出激活图像
% 找到峰值
h = spm_mip_ui('FindMIPax');
% loc     - String defining jump: 'dntmv' - don't move
%                                 'nrvox' - nearest suprathreshold voxel
%                                 'nrmax' - nearest local maxima
%                                 'glmax' - global maxima
loc = 'glmax';
xyz = spm_mip_ui('Jump',h,loc);
```

## 求若干个图像的平均图像
```matlab
function mean_img(input_img,output_img)
% -------------------------------------------------------------------------
% 功能：求若干个图像的平均图像
% 调用：mean_img(input_img,output_img)
% 参数：
%   input_img : cell类型的输入图像，绝对路径
%   output_img : 输出图像的绝对路径
% 示例：
%   input_img = { 'd:\fmri_root\YANTAI\ANALYSIS\mean_smooth4\DW10\s4wraf20161104002-181316-00074-00074-1.img,1'
%                 'd:\fmri_root\YANTAI\ANALYSIS\mean_smooth4\DW11\s4wraf20161104002-181208-00040-00040-1.img,1'
%                 'd:\fmri_root\YANTAI\ANALYSIS\mean_smooth4\DW20\s4wraf20161104002-181822-00227-00227-1.img,1'
%                 'd:\fmri_root\YANTAI\ANALYSIS\mean_smooth4\DW21\s4wraf20161104002-181640-00176-00176-1.img,1'};
%   output_img = 'd:\out.img';
% 	mean_img(input_img,output_img)
% 说明：修改matlabbatch{1}.spm.util.imcalc.expression可以完成不同的计算任务
% -------------------------------------------------------------------------
[path,name,exit] = fileparts(output_img) ;
spm_jobman('initcfg')
%-----------------------------------------------------------------------
% Job configuration created by cfg_util (rev $Rev: 4252 $)
%-----------------------------------------------------------------------
matlabbatch{1}.spm.util.imcalc.input = input_img;
matlabbatch{1}.spm.util.imcalc.output = [name,exit];
matlabbatch{1}.spm.util.imcalc.outdir = {path};
matlabbatch{1}.spm.util.imcalc.expression = '(i1+i2+i3+i4)/4';
matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{1}.spm.util.imcalc.options.mask = 0;
matlabbatch{1}.spm.util.imcalc.options.interp = 1;
matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
%--------------------------------------------------------------------------
spm_jobman('run',matlabbatch);
disp('mean_img successful !');
clear matlabbatch;
```

## 求若干个图像的平均图像(一种速度更快的实现方式)
```matlab
function Q = mean_img2(input_img,output_img)
% -------------------------------------------------------------------------
% 功能：计算四个图像的平均值
% 调用：Q = mean_img2(input_img,output_img)
% 参数：
%   input_img: cell类型的输入图像的绝对路径
%   output_img: 输出图像的绝对路径
% 示例：
% input_img = {'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\s4wraf20160911002-182750-00006-00006-1.img'
%                 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\s4wraf20160911002-182752-00007-00007-1.img'
%                 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\s4wraf20160911002-182754-00008-00008-1.img'
%                 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\s4wraf20160911002-182756-00009-00009-1.img'};
% output_img = 'd:\out.img';
% Q = mean_img2(input_img,output_img)
% -------------------------------------------------------------------------
P = input_img;
Q = output_img;
f = '(i1+i2+i3+i4)/4';
dmtx = 0;
mask = 0;
type = 4;
hold = 0;
flags = {dmtx,mask,type,hold};

[Q,Vo] = spm_imcalc_extend(P,Q,f,flags);
```

##  绘制一个RUN的时间序列
```matlab
function Y = plot_time_series_run(XYZ,run_dir,file_filter)
% -------------------------------------------------------------------------
% 功能 ： 绘制一个RUN的时间序列
% 调用：Y = plot_time_series_run(run_dir,file_filter)
% 参数：
%   XYZ : 图像中的某个点
%   run_dir : run文件夹路径
%   file_filter : 文件过滤器，e.g.w*.img
%   Y：提取的数据
% -------------------------------------------------------------------------
V = cell(272,1);
cd(run_dir);
files = dir(file_filter);
for k = 1:size(files,1)
    V{k} = [run_dir,'\',files(k).name];
end
Y = spm_get_data(V,XYZ);
plot(Y);
```

## 绘制被试的头动图像
```matlab
function plothm(file_path,save_name)
% -------------------------------------------------------------------------
% 功能：根据头动文件画头动图
% 调用：plothm(file_path,save_name)
% 参数：
%   file_path : rp*头动文件绝对路径
%   save_name : 生成图像的名称
% 示例：
%   file_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160716002\ep2d_bold_moco_p2_rest_0016\rp_af20160716002-183602-00006-00006-1.txt';
%   save_name = 'd:\realign_run1.png'
% 	plothm(file_path,save_name)
% 说明：需要事先打开SPM的Graph窗口；save_name如果只有名称，则保存在Matlab当前工作目录下，如果提供了绝对路径，则保存为绝对路径指定的地方。
% -------------------------------------------------------------------------
Params = load(file_path);
fg=spm_figure('FindWin','Graphics');
if ~isempty(fg),
    % display results
    % translation and rotation over time series
    %-------------------------------------------------------------------
    spm_figure('Clear','Graphics');
    ax=axes('Position',[0.1 0.65 0.8 0.2],'Parent',fg,'Visible','off');
    set(get(ax,'Title'),'String','Image realignment','FontSize',16,'FontWeight','Bold','Visible','on');
    ax=axes('Position',[0.1 0.35 0.8 0.2],'Parent',fg,'XGrid','on','YGrid','on');
    plot(Params(:,1:3),'Parent',ax)
    s = ['x translation';'y translation';'z translation'];
    %text([2 2 2], Params(2, 1:3), s, 'Fontsize',10,'Parent',ax)
    legend(ax, s, 0)
    set(get(ax,'Title'),'String','translation','FontSize',16,'FontWeight','Bold');
    set(get(ax,'Xlabel'),'String','image');
    set(get(ax,'Ylabel'),'String','mm');


    ax=axes('Position',[0.1 0.05 0.8 0.2],'Parent',fg,'XGrid','on','YGrid','on');
    plot(Params(:,4:6)*180/pi,'Parent',ax)
    s = ['pitch';'roll ';'yaw  '];
    %text([2 2 2], Params(2, 4:6)*180/pi, s, 'Fontsize',10,'Parent',ax)
    legend(ax, s, 0)
    set(get(ax,'Title'),'String','rotation','FontSize',16,'FontWeight','Bold');
    set(get(ax,'Xlabel'),'String','image');
    set(get(ax,'Ylabel'),'String','degrees');

    % print realigment parameters
    spm_print(save_name);
    print(fg,save_name,'-dpng');% 打印出PNG图片，还可以输出其他的格式，参考Matlab的print函数。
end
return;
```
## 批量保存xjview中的slice_view图像
```matlab
function save_slice_view(file_path,save_path)
%--------------------------------------------------------------------------
% 功能：保存激活图像的slice_view图像
% 调用：save_slice_view(file_path,save_path)
% 参数：
%   file_path : spmT图像
%   save_path : 保存的绝对路径，包含文件名
% 示例：
%   file_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level\20160911002\spmT_0020.hdr';
%   save_path = 'd:\aaa.png';
%   save_slice_view(file_path,save_path)
% -------------------------------------------------------------------------

%file_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level\20160911002\spmT_0020.hdr';
my_xjview(file_path);
hObject = spm_mip_ui('FindMIPax');
eventdata = [];
handles = guidata(hObject);
global sliceview

if(~isfield(sliceview, 'fig') || ~ishandle(sliceview.fig))
    sliceview.viewtype = 't';    
    sliceview.row = 6;
    sliceview.col = 8;
    sliceview.spacing = 4;
    sliceview.page = 1;
    sliceview.data = {{},{},{}}; % t,s,c
    sliceview.slices = {[],[],[]};% t,s,c
    sliceview.colormap = '';
    sliceview.fig = figure('color','k', 'unit','normalized','position',[0.1 0.1 .6 .8],'toolbar','none', 'name', 'xjView slice view', 'NumberTitle','off');
    sliceview.ax = axes('Visible','on','DrawMode','fast','Parent',sliceview.fig,...
    'YDir','normal','Ydir','normal','XTick',[],'YTick',[], 'position', [0.15 0.05 .8 .9]);
    %handles.sliceview.d  = image([],'Tag','Transverse','Parent',handles.sliceview.ax);
    set(sliceview.ax,'XTick',[],'YTick',[]);
    axis equal
    set(sliceview.ax,'color','k');
    %setcolormap(colormp)
    width = 0.05;
    height = 0.025;
    step = 0.025;
    labeloffset = step/2;
end

slicegraph = figure(sliceview.fig);

viewtype =    sliceview.viewtype;   
row =         sliceview.row;
col =         sliceview.col;
spacing =     sliceview.spacing;
page =        sliceview.page;
slice_fig = sliceview.fig;
ax = sliceview.ax;
%d = handles.sliceview.d;

[slicedata, colormp, slices] = cuixu_getSliceViewData(viewtype,row,col, spacing, page);

if isempty(slices)
    return;
end

for ii=1:length(slices)
    if(viewtype == 's')
        postmp = find(slices(ii) - sliceview.slices{2} == 0);
        if(isempty(postmp))
            sliceview.data{2}{end+1} = slicedata{ii};
            sliceview.slices{2}(end+1) = slices(ii);
        end   
    elseif(viewtype == 't')
        postmp = find(slices(ii) - sliceview.slices{1} == 0);
        if(isempty(postmp))
            sliceview.data{1}{end+1} = slicedata{ii};
            sliceview.slices{1}(end+1) = slices(ii);
        end    
    elseif(viewtype == 'c')
        postmp = find(slices(ii) - sliceview.slices{3} == 0);
        if(isempty(postmp))
            sliceview.data{3}{end+1} = slicedata{ii};
            sliceview.slices{3}(end+1) = slices(ii);
        end   
    end  
    
end



%slice_fig = figure('color','k', 'unit','normalized','position',[0.1 0.1 .6 .8],'toolbar','none');


if(length(size(slicedata{1})) == 3)
    [nx, ny, nz] = size(slicedata{1});
    slicedatafinal = zeros(nx*row, ny*col, nz );
    for ii=1:length(slicedata)
        slicedatafinal(nx*(floor((ii-1)/col))+1:nx*(1+floor((ii-1)/col)), ny*(mod(ii-1,col))+1:ny*(mod(ii-1,col)+1), :) = slicedata{ii};
    end
else
    [nx, ny] = size(slicedata{1});
    slicedatafinal = zeros(nx*row, ny*col );
    for ii=1:length(slicedata)
        slicedatafinal(nx*(floor((ii-1)/col))+1:nx*(1+floor((ii-1)/col)), ny*(mod(ii-1,col))+1:ny*(mod(ii-1,col)+1)) = slicedata{ii};
    end
end


try
    delete(handles.sliceview.d)
catch
    [];
end

handles.sliceview.d  = image(slicedatafinal,'Tag','Transverse','Parent',sliceview.ax);

% put slice positions
for ii=1:length(slicedata)
    %text(nx*(floor((ii-1)/col))+1:nx*(1+floor((ii-1)/col)), ny*(mod(ii-1,col))+1:ny*(mod(ii-1,col)+1), num2str(slices(ii)), 'color', 'w');
    text(ny*(mod(ii-1,col))+1, nx*(floor((ii-1)/col))+1+20, num2str(slices(ii)), 'color', 'w');
end
set(sliceview.ax,'XTick',[],'YTick',[]);
axis(sliceview.ax, 'equal');
set(sliceview.ax,'color','k');
guidata(hObject, handles);

%print(handles.figure,'bbb','-dpng');  % 保存主窗口图像
print(slicegraph,save_path,'-dpng');  % 保存slice_view图像
close(slicegraph);
close(handles.figure);
clc;

```

## 获得图像的头信息
```matlab
P = {'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\af20160911002-182750-00006-00006-1.img',
    'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\20160911002\ep2d_bold_moco_p2_rest_0006\af20160911002-182750-00006-00006-1.img'};
header = spm_vol(P);
```

## 获得图像某个坐标的值
```matlab
% 获得图像某个坐标的值
V = {'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level\20160911002\beta_0001.img'};
XYZ = [13;48;2];
Y = spm_get_data(V,XYZ);
```

## 批量做图像的平滑处理
```matlab 
function smooth(filter,pres)
% -------------------------------------------------------------------------
% 功能： 平滑图像
% 调用：smooth(filter,pres)
% 参数：
%   filter : 控制处理的被试数量，例如'20161001*';
%   pres : 生成的平滑之后的图像的前缀，例如's';
% 示例：
%   filter = '20161003*';
%   pres = 's';
%   smooth(filter,pres);
% 说明：平滑之后的图像与输入图像在同一文件夹，前缀不同
% -------------------------------------------------------------------------
clc;
warning('off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%需要修改的变量，修改此处可以处理多个被试的数据
%filter = '2016*';
% 输入： wraf*开头的图像文件
% 输出： 平滑之后的文件(与输入文件在同一文件夹中，前缀不一样)
%-------------------------------
% 配置参数：
% filter : 控制处理的被试数量
% pres : 输出图像的前缀
% pre_processing_path ： wraf*开头的图像文件的绝对路径
% run_num ： 每个RUN的文件数量(删除TR之后的数量)
% runExpID : RUN文件夹名称，这里是'ep2d*'
% filenames = dir('wraf*.img'); 输入图像以wraf开头
% matlabbatch{1}.spm.spatial.smooth.fwhm = [4 4 4]; 控制平滑核大小
% matlabbatch{1}.spm.spatial.smooth.prefix = 's4';输出文件的前缀
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%预处理文件夹，整个预处理过程保存在这里
pre_processing_path = {'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\'};
%删除空TR后每个run文件的数量
run_num = 272;
%开始处理    
cd(pre_processing_path{1});
subExpID = dir(filter); %====================================
for i=1:size(subExpID,1)
    cd([pre_processing_path{1},subExpID(i).name]);
    
	diary smooth_output.txt; % 重定向控制台输出到文件
	tic;  %开始计时   
    %2.获得run的功能像文件
    cd([pre_processing_path{1},subExpID(i).name]);
    runExpID=dir('ep2d*');
    for j=1:size(runExpID,1)
        cd ([pre_processing_path{1},subExpID(i).name,'\',runExpID(j).name]);
        filenames = dir('wraf*.img');
        funcFilenames = cell(run_num,1);%每个run的文件集合
        for k=1:size(filenames,1)
            funcFilenames{k} = [pre_processing_path{1},subExpID(i).name,'\',runExpID(j).name,'\',filenames(k).name,',1'];
        end
         
        funcFilenames = {funcFilenames};
        
        %%================================batch-begin===================================================%%
        spm_jobman('initcfg')
        
        matlabbatch{1}.spm.spatial.smooth.data = funcFilenames{1};
        matlabbatch{1}.spm.spatial.smooth.fwhm = [4 4 4];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = pres;
        %%================================batch-end===================================================%%
        spm_jobman('run',matlabbatch);
        disp('smooth successful !');
        clear matlabbatch
    end
	toc
	diary off ;
end

```

## 图像计算函数的拓展(针对fMRI图像)
```matlab
function [Q,Vo] = spm_imcalc_extend(P,Q,f,flags,varargin)
% Perform algebraic functions on images
% FORMAT Q = spm_imcalc_ui(P,Q,f,flags)
% P             - matrix of input image filenames
%                 [user prompted to select files if arg missing or empty]
% Q             - name of output image
%                 [user prompted to enter filename if arg missing or empty]
% f             - expression to be evaluated
%                 [user prompted to enter expression if arg missing or empty]
% flags         - cell vector of flags: {dmtx,mask,type,hold}
% dmtx          - Read images into data matrix?
%                 [defaults (missing or empty) to 0 - no]
% mask          - implicit zero mask?
%                 [defaults (missing or empty) to 0]
% type          - data type for output image (see spm_type)
%                 [defaults (missing or empty) to 4 - 16 bit signed shorts]
% hold          - interpolation hold (see spm_slice_vol)
%                 [defaults (missing or empty) to 0 - nearest neighbour]
% Q (output)    - full pathname of image written
% Vo            - structure containing information on output image (see spm_vol)
%
%_______________________________________________________________________
%
% spm_imcalc_ui uses spm_imcalc as an engine to perform user-specified
% algebraic manipulations on a set of images, with the result being
% written out as an image. The user is prompted to supply images to
% work on, a filename for the output image, and the expression to
% evaluate. The expression should be a standard matlab expression,
% within which the images should be referred to as i1, i2, i3,... etc.
%
% If the dmtx flag is set, then images are read into a data matrix X
% (rather than into seperate variables i1, i2, i3,...). The data matrix
% should be referred to as X, and contains images in rows.
%
% Computation is plane by plane, so in data-matrix mode, X is a NxK
% matrix, where N is the number of input images [prod(size(Vi))], and K
% is the number of voxels per plane [prod(Vi(1).dim(1:2))].
%
% For data types without a representation of NaN, implicit zero masking
% assummes that all zero voxels are to be treated as missing, and
% treats them as NaN. NaN's are written as zero (by spm_write_plane),
% for data types without a representation of NaN.
%
% With images of different sizes and orientations, the size and
% orientation of the first is used for the output image. A warning is
% given in this situation. Images are sampled into this orientation
% using the interpolation specified by the hold parameter.  [default -
% nearest neighbour]
%
% The image Q is written to current working directory unless a valid
% full pathname is given.
%
% Example expressions (f):
%
%        i) Mean of six images (select six images)
%           f = '(i1+i2+i3+i4+i5+i6)/6'
%       ii) Make a binary mask image at threshold of 100
%           f = 'i1>100'
%      iii) Make a mask from one image and apply to another
%           f = 'i2.*(i1>100)'
%                 - here the first image is used to make the mask, which is
%                   applied to the second image
%       iv) Sum of n images
%           f = 'i1 + i2 + i3 + i4 + i5 + ...'
%        v) Sum of n images (when reading data into a data-matrix - use dmtx arg)
%           f = 'sum(X)'
% 
% Parameters can be passed as arguments to override internal defaults
% (for hold, mask & type), or to pre-specify images (P), output
% filename (Q), or expression (f). Pass empty matrices for arguments
% not to be set.
% E.g.  Q = spm_imcalc_ui({},'test','',{[],[],[],1})
%       ... pre-specifies the output filename as 'test.img' in the current
% working directory, and sets the interpolation hold to tri-linear.
%
% Further, if calling spm_imcalc directly, additional variables for use in
% the computation can be passed at the end of the argument list. These
% should be referred to by the names of the arguments passed in the
% expression to be evaluated. E.g. if c is a 1xn vector of weights, then
% for n images, using the (dmtx) data-matrix version, the weighted sum can
% be computed using:
%       Vi= spm_vol(spm_select(inf,'image'));
%       Vo= Vi(1);
%       Vo.fname = 'output.img';
%       Vo.pinfo(1:2) = Inf;
%       Q = spm_imcalc(Vi,Vo,'c*X',{1},c)
% Here we've pre-specified the expression and passed the vector c as an
% additional variable (you'll be prompted to select the n images).
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% John Ashburner & Andrew Holmes
% $Id: spm_imcalc_ui.m 3691 2010-01-20 17:08:30Z guillaume $

%-GUI setup
%--------------------------------------------------------------------------
SVNid = '$Rev: 3691 $';
%[Finter,Fgraph,CmdLine] = spm('FnUIsetup','ImCalc',0);
spm('FnBanner',mfilename,SVNid);
%-Condition arguments
%--------------------------------------------------------------------------
if nargin<4, flags={}; end
if nargin<3, f=''; end
if nargin<2, Q=''; end
if nargin<1, P={}; end

%if isempty(P), P = %spm_select(Inf,'image','Select images to work on'); end
if isempty(P), error('no input images specified'), end
%if isempty(Q), Q = %spm_input('Output filename',1,'s'); end
if isempty(f), f = spm_input('Evaluated Function',2,'s'); end

if length(flags)<4, hold=[]; else hold=flags{4}; end
if isempty(hold), hold=0; end
if length(flags)<3, type=[]; else type=flags{3}; end
if isempty(type), type=4; end, if ischar(type), type=spm_type(type); end
if length(flags)<2, mask=[]; else mask=flags{2}; end
if isempty(mask), mask=0; end
if length(flags)<1, dmtx=[]; else dmtx=flags{1}; end
if isempty(dmtx), dmtx=0; end

%spm('FigName','ImCalc: working',Finter,CmdLine);
%spm('Pointer','Watch')

%-Map input files
%--------------------------------------------------------------------------
Vi = spm_vol(char(P));
if isempty(Vi), error('no input images specified'), end

%-Check for consistency of image dimensions and orientation / voxel size
%--------------------------------------------------------------------------
if length(Vi)>1 && any(any(diff(cat(1,Vi.dim),1,1),1))
    warning(['images don''t all have same dimensions',...
        ' - using those of 1st image']);
end
if any(any(any(diff(cat(3,Vi.mat),1,3),3)))
    warning(['images don''t all have same orientation & voxel size',...
        ' - using 1st image']);
end

%-Work out filename for output image
%--------------------------------------------------------------------------
[p n e] = spm_fileparts(Q);
if isempty(p), p = pwd; end
if ~exist(p,'dir')
    warning('Invalid directory: writing to current directory')
    p = pwd;
end

Vo = struct('fname',   fullfile(p, [n, e]),...
            'dim',     Vi(1).dim(1:3),...
            'dt',      [type spm_platform('bigend')],...
            'mat',     Vi(1).mat,...
            'descrip', 'spm - algebra');

%-Call spm_imcalc to handle computations
%--------------------------------------------------------------------------
args = {dmtx,mask,hold};
Vo   = spm_imcalc(Vi,Vo,f,args);

%-End
%--------------------------------------------------------------------------
%spm('Pointer');
%spm('FigName','ImCalc: done',Finter,CmdLine);

```
