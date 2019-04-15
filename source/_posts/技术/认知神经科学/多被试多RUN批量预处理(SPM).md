---
title: 多被试多RUN批量预处理(SPM)
toc: true

tags:
  - fMIRI
date: 2017-03-20 20:50:00
---

使用SPM处理多个被试多个RUN的数据。代码涉及到的文件夹结构如下：
<!-- more -->

├─20160911002  被试文件夹
│  ├─ep2d_bold_moco_p2_rest_0006 第一个RUN
│  ├─ep2d_bold_moco_p2_rest_0011 第二个RUN
│  ├─ep2d_bold_moco_p2_rest_0016 第三个RUN
│  ├─ep2d_bold_moco_p2_rest_0021 第四个RUN
│  └─t1_mprage_sag_p2_0026 被试的结构像
│      ├─run1
│      ├─run2
│      ├─run3
│      └─run4
├─20160916001
│  ├─ep2d_bold_moco_p2_rest_0006
│  ├─ep2d_bold_moco_p2_rest_0011
│  ├─ep2d_bold_moco_p2_rest_0016
│  ├─ep2d_bold_moco_p2_rest_0021
│  └─t1_mprage_sag_p2_0026
│      ├─run1
│      ├─run2
│      ├─run3
│      └─run4

```matlab
function pre_processing(filter)
% -------------------------------------------------------------------------
% 功能： 数据预处理，包括 1.slice timing; 2. realign； 3.配准；4.分割；5.标准化；6.平滑
% 调用： pre_processing(filter)
% 参数：
%   filter:控制处理的被试，例如'20161001*'    
% 示例：
%   filter = ‘20161001*’;
%   pre_processing(filter);
% 说明：输入图像需要满足预定的文件夹结构，该结构为使用SPM8 Batch进行格式转换默认生成的结构
%--------------------------------------------------------------------------
clc;
warning('off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%需要修改的变量，修改此处可以处理多个被试的数据
%filter = '20161024002*';
% 输入： 经过格式转换之后的img,hdr图像
% 输出： 预处理之后的图像和预处理的结果文件

% 配置参数：
% filter : 控制处理的被试数量
% img_hdr_path ：img,hdr文件存放的绝对路径
% pre_processing_path ： wraf*开头的图像文件的绝对路径
% delete_filenameID ：预处理开始之前需要删除的TR。
% run_num ： 每个RUN的文件数量(删除TR之后的数量)
% innerMatlab_path ： SPM8工具箱中灰质，白质，脑脊液文件的路径，这个在分割的时候会用到，在不同电脑间移植的时候需要修改。
% runExpID : RUN文件夹名称，这里是'ep2d*'
% filenames = dir('wraf*.img'); 输入图像以wraf开头
% matlabbatch{1}.spm.spatial.smooth.fwhm = [4 4 4]; 控制平滑核大小
% matlabbatch{1}.spm.spatial.smooth.prefix = 's4';输出文件的前缀
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%转换完格式的文件存放的文件夹
img_hdr_path = {'D:\FMRI_ROOT\YANTAI\ANALYSIS\img_hdr\'};
%预处理文件夹，整个预处理过程保存在这里
pre_processing_path = {'D:\FMRI_ROOT\YANTAI\ANALYSIS\pre_processing\'};
%要删除掉的TR
delete_filenameID = {'*-00001-00001-*','*-00002-00002-*','*-00003-00003-*','*-00004-00004-*','*-00005-00005-*'};
%删除空TR后每个run文件的数量
run_num = 272;
%内部路径，根据SPM8安装路径修改
innerMatlab_path = {
                   'C:\mazcx\matlabtoolbox\spm8\tpm\csf.nii,1'
                   'C:\mazcx\matlabtoolbox\spm8\tpm\grey.nii,1'
                   'C:\mazcx\matlabtoolbox\spm8\tpm\white.nii,1'
                    };
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%预处理部分，主要内容：
%1. 将格式转换后的文件复制一份到预处理文件夹
%2. 删除掉多余的TR，只剩下需要的TR
%3. 在结构像文件中新建4个run，把结构像复制到每个run下,配准的时候每个run都是用自己对应run的结构像
%4. 以每个run为单位，运行预处理的batch文件
%5. 控制处理的文件夹主要需要修改的变量：
%  把代码中带=============注释的改成被试文件夹名称
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%将数据拷贝到预处理文件夹
cd(img_hdr_path{1});
copyfile(filter,pre_processing_path{1}); %====================

%删除多余的TR
cd(pre_processing_path{1});
subExpID=dir (filter);     %被试文件夹==================

for i=1:size(subExpID,1)
     cd ([pre_processing_path{1},subExpID(i).name]);
     runExpID = dir('ep2d*'); %run 文件夹++++++++++++++++++++++++
     for j=1:size(runExpID,1)
         cd ([pre_processing_path{1},subExpID(i).name,'\',runExpID(j).name]);
         for k = 1:size(delete_filenameID,2)
             delete(cell2mat(delete_filenameID(k)));
         end
     end
end

%复制4个run的结构像
cd(pre_processing_path{1});
subExpID = dir(filter);   %=========================
for i=1:size(subExpID,1)
    cd([pre_processing_path{1},subExpID(i).name]);
    t1ExpID = dir('t1*');
    cd(t1ExpID.name);
    mkdir('run1');
    copyfile('s*',[pre_processing_path{1},subExpID(i).name,'\',t1ExpID.name,'\run1\']);
    mkdir('run2');
    copyfile('s*',[pre_processing_path{1},subExpID(i).name,'\',t1ExpID.name,'\run2\']);
    mkdir('run3');
    copyfile('s*',[pre_processing_path{1},subExpID(i).name,'\',t1ExpID.name,'\run3\']);
    mkdir('run4');
    copyfile('s*',[pre_processing_path{1},subExpID(i).name,'\',t1ExpID.name,'\run4\']);
end

%开始预处理    
cd(pre_processing_path{1});
subExpID = dir(filter); %====================================
for i=1:size(subExpID,1)
    cd([pre_processing_path{1},subExpID(i).name]);
    
	diary pre_processing_output.txt; % 重定向控制台输出到文件
	tic;  %开始计时   
    %1. 获得4个结构像文件
    t1ExpID = dir('t1*');
    cd([pre_processing_path{1},subExpID(i).name,'\',t1ExpID(1).name]);%切换到t1像
    runID = dir('run*');
    data3D_filenames=cell(4,1);%4个run的结构像文件
    for j=1:size(runID,1)
        cd([pre_processing_path{1},subExpID(i).name,'\',t1ExpID(1).name,'\',runID(j).name])
        filenames = dir('s*.img');
        data3D_filenames{j} = [pre_processing_path{1},subExpID(i).name,'\',t1ExpID(1).name,'\',runID(j).name,'\',filenames(1).name,',1'];
    end
    
    %2.获得run的功能像文件
    cd([pre_processing_path{1},subExpID(i).name]);
    runExpID=dir('ep2d*');
    for j=1:size(runExpID,1)
        cd ([pre_processing_path{1},subExpID(i).name,'\',runExpID(j).name]);
        filenames = dir('f*.img');
        funcFilenames = cell(run_num,1);%每个run的文件集合
        for k=1:size(filenames,1)
            funcFilenames{k} = [pre_processing_path{1},subExpID(i).name,'\',runExpID(j).name,'\',filenames(k).name,',1'];
        end
         
        funcFilenames = {funcFilenames};
        data3D_filename = { data3D_filenames{j} };
        
        %%================================batch-begin===================================================%%
        spm_jobman('initcfg')
        matlabbatch{1}.spm.temporal.st.scans = funcFilenames;
        matlabbatch{1}.spm.temporal.st.nslices = 33;
        matlabbatch{1}.spm.temporal.st.tr = 2;
        matlabbatch{1}.spm.temporal.st.ta = 1.93939393939394;
        matlabbatch{1}.spm.temporal.st.so = [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32];
        matlabbatch{1}.spm.temporal.st.refslice = 33;
        matlabbatch{1}.spm.temporal.st.prefix = 'a';
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1) = cfg_dep;
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).tname = 'Session';
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).tgt_spec{1}(1).name = 'filter';
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).tgt_spec{1}(1).value = 'image';
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).tgt_spec{1}(2).name = 'strtype';
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).tgt_spec{1}(2).value = 'e';
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).sname = 'Slice Timing: Slice Timing Corr. Images (Sess 1)';
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).src_exbranch = substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1});
        matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1).src_output = substruct('()',{1}, '.','files');
        matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
        matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.sep = 4;
        matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
        matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
        matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.interp = 2;
        matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
        matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.weight = '';
        matlabbatch{2}.spm.spatial.realign.estwrite.roptions.which = [2 1];
        matlabbatch{2}.spm.spatial.realign.estwrite.roptions.interp = 4;
        matlabbatch{2}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
        matlabbatch{2}.spm.spatial.realign.estwrite.roptions.mask = 1;
        matlabbatch{2}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1) = cfg_dep;
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).tname = 'Reference Image';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).tgt_spec{1}(1).name = 'filter';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).tgt_spec{1}(1).value = 'image';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).tgt_spec{1}(2).name = 'strtype';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).tgt_spec{1}(2).value = 'e';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).sname = 'Realign: Estimate & Reslice: Mean Image';
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).src_exbranch = substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1});
        matlabbatch{3}.spm.spatial.coreg.estimate.ref(1).src_output = substruct('.','rmean');
        matlabbatch{3}.spm.spatial.coreg.estimate.source = data3D_filename; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        matlabbatch{3}.spm.spatial.coreg.estimate.other = {''};
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
        matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
        matlabbatch{4}.spm.spatial.preproc.data = data3D_filename ; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        matlabbatch{4}.spm.spatial.preproc.output.GM = [0 0 1];
        matlabbatch{4}.spm.spatial.preproc.output.WM = [0 0 1];
        matlabbatch{4}.spm.spatial.preproc.output.CSF = [0 0 1];
        matlabbatch{4}.spm.spatial.preproc.output.biascor = 1;
        matlabbatch{4}.spm.spatial.preproc.output.cleanup = 1;
        matlabbatch{4}.spm.spatial.preproc.opts.tpm = innerMatlab_path; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        matlabbatch{4}.spm.spatial.preproc.opts.ngaus = [2
                                                         2
                                                         2
                                                         4];
        matlabbatch{4}.spm.spatial.preproc.opts.regtype = 'mni';
        matlabbatch{4}.spm.spatial.preproc.opts.warpreg = 1;
        matlabbatch{4}.spm.spatial.preproc.opts.warpco = 25;
        matlabbatch{4}.spm.spatial.preproc.opts.biasreg = 0.0001;
        matlabbatch{4}.spm.spatial.preproc.opts.biasfwhm = 60;
        matlabbatch{4}.spm.spatial.preproc.opts.samp = 3;
        matlabbatch{4}.spm.spatial.preproc.opts.msk = {''};
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1) = cfg_dep;
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).tname = 'Parameter File';
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).tgt_spec{1}(1).name = 'filter';
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).tgt_spec{1}(1).value = 'mat';
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).tgt_spec{1}(2).name = 'strtype';
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).tgt_spec{1}(2).value = 'e';
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).sname = 'Segment: Norm Params Subj->MNI';
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).src_exbranch = substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1});
        matlabbatch{5}.spm.spatial.normalise.write.subj.matname(1).src_output = substruct('()',{1}, '.','snfile', '()',{':'});
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep;
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).tname = 'Images to Write';
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).tgt_spec{1}(1).name = 'filter';
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).tgt_spec{1}(1).value = 'image';
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).tgt_spec{1}(2).name = 'strtype';
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).tgt_spec{1}(2).value = 'e';
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).sname = 'Realign: Estimate & Reslice: Resliced Images (Sess 1)';
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).src_exbranch = substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1});
        matlabbatch{5}.spm.spatial.normalise.write.subj.resample(1).src_output = substruct('.','sess', '()',{1}, '.','rfiles');
        matlabbatch{5}.spm.spatial.normalise.write.roptions.preserve = 0;
        matlabbatch{5}.spm.spatial.normalise.write.roptions.bb = [-90 -126 -72
                                                                  90 90 108];
        matlabbatch{5}.spm.spatial.normalise.write.roptions.vox = [3 3 3];
        matlabbatch{5}.spm.spatial.normalise.write.roptions.interp = 1;
        matlabbatch{5}.spm.spatial.normalise.write.roptions.wrap = [0 0 0];
        matlabbatch{5}.spm.spatial.normalise.write.roptions.prefix = 'w';
        matlabbatch{6}.spm.spatial.smooth.data(1) = cfg_dep;
        matlabbatch{6}.spm.spatial.smooth.data(1).tname = 'Images to Smooth';
        matlabbatch{6}.spm.spatial.smooth.data(1).tgt_spec{1}(1).name = 'filter';
        matlabbatch{6}.spm.spatial.smooth.data(1).tgt_spec{1}(1).value = 'image';
        matlabbatch{6}.spm.spatial.smooth.data(1).tgt_spec{1}(2).name = 'strtype';
        matlabbatch{6}.spm.spatial.smooth.data(1).tgt_spec{1}(2).value = 'e';
        matlabbatch{6}.spm.spatial.smooth.data(1).sname = 'Normalise: Write: Normalised Images (Subj 1)';
        matlabbatch{6}.spm.spatial.smooth.data(1).src_exbranch = substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1});
        matlabbatch{6}.spm.spatial.smooth.data(1).src_output = substruct('()',{1}, '.','files');
        matlabbatch{6}.spm.spatial.smooth.fwhm = [6 6 6];
        matlabbatch{6}.spm.spatial.smooth.dtype = 0;
        matlabbatch{6}.spm.spatial.smooth.im = 0;
        matlabbatch{6}.spm.spatial.smooth.prefix = 's';
        %%================================batch-end===================================================%%
        spm_jobman('run',matlabbatch);
        disp('pre_processing successful !');
        clear matlabbatch
    end
	toc
	diary off ;
end



```