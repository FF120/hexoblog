---
title: 动态因果模型(DCM)的批量定义和估计
toc: true

tags:
  - fMRI
date: 2017-03-20 21:19:23
---
借助SPM中的DCM，实现批量定义模型和批量估计。附带一些批量查看和保存结果的函数。
重写了SPM中的一些函数，带extend的为重写的函数。
<!-- more -->

## spm_dcm_specify_extend.m
```matlab
function DCM = spm_dcm_specify_extend(spmmatfile,name,VOIs,condition_mask,TE,Input_a,Input_b,Input_c)
% Specify inputs of a DCM
% FORMAT [DCM] = spm_dcm_specify
%
% DCM  - the DCM structure (see spm_dcm_ui)
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_specify.m 4185 2011-02-01 18:46:18Z guillaume $


%-Interactive window
%--------------------------------------------------------------------------
Finter = spm_figure('GetWin','Interactive');
bcolor = get(Finter,'color');
WS     = spm('WinScale');
dx     = 20;

spm_input('Specify DCM:...  ',1,'d');

%==========================================================================
% Get design and directory
%==========================================================================
%[spmmatfile, sts] = spm_select(1,'^SPM\.mat$','Select SPM.mat');
%spmmatfile = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\20160716002\SPM.mat';
sts = 1;
if ~sts, DCM = []; return; end
swd = spm_str_manip(spmmatfile,'H');
try
    load(fullfile(swd,'SPM.mat'))
catch
    error(['Cannot read ' fullfile(swd,'SPM.mat')]);
end


%==========================================================================
% Name
%==========================================================================
%name  = spm_input('name for DCM_???.mat','+1','s');
%name = 'HG_DCM';

%==========================================================================
% Outputs
%==========================================================================

%-Get cell array of region structures
%--------------------------------------------------------------------------
% VOI1_path = 'd:\fmri_root\yantai\aNALYSIS\first_level_dcm_w_whole\20160716002\VOI_HG_DW_1.mat';
% VOI2_path = 'd:\fmri_root\yantai\aNALYSIS\first_level_dcm_w_whole\20160716002\VOI_MFG_DW_1.mat';
% VOI3_path = 'd:\fmri_root\yantai\aNALYSIS\first_level_dcm_w_whole\20160716002\VOI_STG_DW_1.mat';
% VOIs = {VOI1_path;VOI2_path;VOI3_path};
%P     = cellstr(spm_select([1 8],'^VOI.*\.mat$',{'select VOIs'},'',swd));
P = VOIs;
m     = numel(P);
for i = 1:m
    p     = load(P{i},'xY');
    xY(i) = p.xY;
end


%==========================================================================
% Inputs
%==========================================================================

%-Get (nc) 'causes' or inputs U
%--------------------------------------------------------------------------
spm_input('Input specification:...  ',1,'d');
Sess   = SPM.Sess(xY(1).Sess);
%condition_mask = [1,0,0,0]; % 配置包含哪个条件，不包含哪个条件＿
if isempty(Sess.U)
    % spontaneous activity, i.e. no stimuli
    nc = 0;
    U = [];
else
    % with stimuli
    U.dt   = Sess.U(1).dt;
    u      = length(Sess.U);
    U.name = {};
    U.u    = [];
    for  i = 1:u  %i 是条件的编号  1＿JX＿ 2＿DW   3＿RL 4＿ZR
        for j = 1:length(Sess.U(i).name)
            str = ['include ' Sess.U(i).name{j} '?'];  % include JX
            if condition_mask(i)%spm_input(str,'+1','y/n',[1 0],1)   include JX20 ,yes 1 ; no 0;
                U.u             = [U.u Sess.U(i).u(33:end,j)];
                U.name{end + 1} = Sess.U(i).name{j};
            end
        end
    end
    nc     = size(U.u,2);
end

%==========================================================================
% Timings
%==========================================================================

spm_input('Timing information:...  ',-1,'d');

%-Slice timings
%--------------------------------------------------------------------------
RT     = SPM.xY.RT;
%delays = spm_input('Slice timings [s]','+1','r', repmat(RT,1,m),m,[0 RT]);
delays = repmat(RT,1,m)';
%-Echo time (TE) of data acquisition
%--------------------------------------------------------------------------
%TE    = 0.04;  %==================================================================自己输入TE的忽===================
TE_ok = 0;
while ~TE_ok
    %TE = spm_input('Echo time, TE [s]', '+1', 'r', TE);
    if ~TE || (TE < 0) || (TE > 0.1)
        str = { 'Extreme value for TE or TE undefined.',...
            'Please re-enter TE (in seconds!)'};
        spm_input(str,'+1','bd','OK',[1],1);
    else
        TE_ok = 1;
    end
end


%==========================================================================
% Model options
%==========================================================================
if nc                                                     % there are inputs
    spm_input('Model options:...  ',-1,'d');
    %options.nonlinear  = spm_input('modulatory effects','+1','b',{'bilinear','nonlinear'},[0 1],1);  
    options.nonlinear = 0; %modulatory effects :  0: bilinear  , 1: nonlinear
	%options.two_state  = spm_input('states per region', '+1','b',{'one','two'},[0 1],1);
	options.two_state = 0 ; % states per region ; 0: one  ; 1 : two
    %options.stochastic = spm_input('stochastic effects','+1','b',{'no','yes'},[0 1],1);
	options.stochastic = 0 ; %stochastic effects; 0 : no ; 1 : yes,
    %options.centre     = spm_input('centre input',      '+1','b',{'no','yes'},[0 1],1);
	options.centre = 0 ; % centre input ; 0 : no 1: yes;
    options.endogenous = 0;
else
    options.nonlinear  = 0;
    options.two_state  = 0;
    options.stochastic = 1;
    options.centre     = 1;
    options.endogenous = 1;
end

%==========================================================================
% Graph connections
%==========================================================================
a     = zeros(m,m);
if options.endogenous
    b     = zeros(m,m,1);
    c     = zeros(m,1);
else
    b     = zeros(m,m,nc);
    c     = zeros(m,nc);
end
d     = zeros(m,m,0);

%-Intrinsic connections (A matrix)
%==========================================================================

%-Buttons and labels
%--------------------------------------------------------------------------
spm_input('Specify intrinsic connections from',1,'d')
spm_input('to',3,'d')
for i = 1:m
    str    = sprintf('%s %i',xY(i).name,i);
    h1(i)  = uicontrol(Finter,'String',str,...
        'Style','text',...
        'FontSize',10,...
        'BackgroundColor',bcolor,...
        'HorizontalAlignment','right',...
        'Position',[080 350-dx*i 080 020].*WS);
    h2(i)  = uicontrol(Finter,'String',sprintf('%i',i),...
        'Style','text',...
        'FontSize',10,...
        'BackgroundColor',bcolor,...
        'Position',[180+dx*i 350 010 020].*WS);
end
for i = 1:m
    for j = 1:m
        h3(i,j) = uicontrol(Finter,...
            'Position',[180+dx*j 350-dx*i 020 020].*WS,...
            'BackgroundColor',bcolor,...
            'Style','radiobutton');
        if i == j
            set(h3(i,j),'Value',1,...
                'enable','off');
        else
            set(h3(i,j),'enable','on','TooltipString', ...
                sprintf('from %s to %s',xY(j).name,xY(i).name));
        end
        if nc && i~=j
            set(h3(i,j),'Value',0);
        else
            set(h3(i,j),'Value',1);
        end
    end
end
uicontrol(Finter,'String','done','Position', [300 100 060 020].*WS,...
    'Callback', 'uiresume(gcbf)');

%uiwait(Finter);

%-Get a  a 就是DCM模型的连接矩阿
%--------------------------------------------------------------------------
%for i = 1:m
%    for j = 1:m
%       a(i,j) = get(h3(i,j),'Value');
%   end
%end
%a = [1,1,1;1,1,1;1,1,1];  %=========================================================定义的DCM模型，此处有三个节点，所以是3*3的矩阵；
a = Input_a;
delete(findobj(get(Finter,'Children'),'flat'));

%-Effects of causes (B and C matrices)
%==========================================================================
uicontrol(Finter,'String','done','Position', [300 100 060 020].*WS,...
    'Callback', 'uiresume(gcbf)');
for k = 1:nc

    %-Buttons and labels
    %----------------------------------------------------------------------
    str   = sprintf(...
        'Effects of %-12s on regions... and connections',...
        U.name{k});
    spm_input(str,1,'d');      % Effects of JX           on regions... and connections

    for i = 1:m
        h1(i)  = uicontrol(Finter,'String',xY(i).name,...
            'Style','text',...
            'BackgroundColor',bcolor,...
            'FontSize',10,...
            'Position',[080 350-dx*i 080 020].*WS);
        h2(i)  = uicontrol(Finter,...
            'Position',[160 360-dx*i 020 020].*WS,...
            'BackgroundColor',bcolor,...
            'Style','radiobutton');
    end
    for i = 1:m
        for j = 1:m
            if a(i,j) == 1

                % Allow modulation of intrinsic connections
                %----------------------------------------------------------
                h3(i,j) = uicontrol(Finter,...
                    'Position',[220+dx*j 360-dx*i 020 020].*WS,...
                    'BackgroundColor',bcolor,...
                    'Style','radiobutton');
                set(h3(i,j),'TooltipString', ...
                    sprintf('from %s to %s',xY(j).name,xY(i).name));

            end
        end
    end

    %uiwait(Finter);

    %-Get c   就是左边的第丿?，代表输入加在那个脑区上＿ * 1
    %----------------------------------------------------------------------
    %for i = 1:m
        %c(i,k)   = get(h2(i),'Value');
    %end
	%c = [1,0,0];
    c = Input_c;
    %-Get b allowing any 2nd order effects   3*3 的矩阵，代表调节变量在哪条线上??里有三个区域，承?昿*3
    %----------------------------------------------------------------------
    % for i = 1:m
        % for j = 1:m
            % if a(i,j)==1
                % b(i,j,k) = get(h3(i,j),'Value');
            % end
        % end
    % end
	%b = [0,0,0;0,0,0;0,0,0];
    b = Input_b;
    delete([h1(:); h2(:); h3(a==1)])

end
delete(findobj(get(Finter,'Children'),'flat'));


%-Effects of nonlinear modulations (D matrices)
%==========================================================================
if options.nonlinear
    uicontrol(Finter,'String','done','Position', [300 100 060 020].*WS,...
        'Callback', 'uiresume(gcbf)');
    for k = 1:m

        %-Buttons and labels
        %------------------------------------------------------------------
        str = sprintf('Effects of %-12s activity on connections',xY(k).name);
        spm_input(str,1,'d');

        for i = 1:m
            for j = 1:m
                if a(i,j)==1

                    % Allow modulation of intrinsic connections
                    %------------------------------------------------------
                    h4(i,j) = uicontrol(Finter,...
                        'Position',[220+dx*j 360-dx*i 020 020].*WS,...
                        'BackgroundColor',bcolor,...
                        'Style','radiobutton');
                end
            end
        end

        uiwait(Finter);

        %-Get d allowing any 2nd order effects
        %------------------------------------------------------------------
        for i = 1:m
            for j = 1:m
                if a(i,j)==1
                    d(i,j,k) = get(h4(i,j),'Value');
                end
            end
        end
        delete(h4(a==1))

    end
end

delete(findobj(get(Finter,'Children'),'flat'));
spm_input('Thank you',1,'d')


%==========================================================================
% Response
%==========================================================================

%-Response variables & confounds (NB: the data have been whitened)
%--------------------------------------------------------------------------
n     = length(xY);                      % number of regions
v     = length(xY(1).u);                 % number of time points
Y.dt  = SPM.xY.RT;
Y.X0  = xY(1).X0;
for i = 1:n
    Y.y(:,i)  = xY(i).u;
    Y.name{i} = xY(i).name;
end

%-Error precision components (one for each region) - i.i.d. (because of W)
%--------------------------------------------------------------------------
Y.Q        = spm_Ce(ones(1,n)*v);


%==========================================================================
% DCM structure
%==========================================================================

% Endogenous input specification
if isempty(U)
    U.u    = zeros(v,1);
    U.name = {'null'};
end

%-Store all variables in DCM structure
%--------------------------------------------------------------------------
DCM.a       = a;
DCM.b       = b;
DCM.c       = c;
DCM.d       = d;
DCM.U       = U;
DCM.Y       = Y;
DCM.xY      = xY;
DCM.v       = v;
DCM.n       = n;
DCM.TE      = TE;
DCM.delays  = delays;
DCM.options = options;

%-Save
%--------------------------------------------------------------------------
if spm_check_version('matlab','7') >= 0
    save(fullfile(swd,['DCM_' name '.mat']),'-V6','DCM');
else
    save(fullfile(swd,['DCM_' name '.mat']),'DCM');
end

```

## spm_getSPM_extend.m
``` matlab
function [SPM,xSPM] = spm_getSPM_extend(varargin)
% Compute a specified and thresholded SPM/PPM following estimation
% FORMAT [SPM,xSPM] = spm_getSPM;
% Query SPM in interactive mode.
%
% FORMAT [SPM,xSPM] = spm_getSPM(xSPM);
% Query SPM in batch mode. See below for a description of fields that may
% be present in xSPM input. Values for missing fields will be queried
% interactively.

%FORMAT [SPM,xSPM] = spm_getSPM(spm_mat_path,InputIC,InputMask，InputthresDesc，Input_u);
% spm_mat_path : SPM.mat path
% InputIC : 选择设置的第几个Contrast,整数＿
% InputMask;  %  0 none  1 contrast  2 image ============
% InputthresDesc ;  %===p value adjustment to control: 'FWE' or 'none'==
% Input_u; %threshold  默认0.001====

%-GUI setup
%--------------------------------------------------------------------------
spm_help('!ContextHelp',mfilename)
spm('Pointer','Arrow')

%-Select SPM.mat & note SPM results directory
%--------------------------------------------------------------------------
if nargin == 1
    xSPM = varargin{1};
end
if nargin > 1
    spm_mat_path = varargin{1};
	InputIC = varargin{2};
    InputMask = varargin{3};
    InputthresDesc = varargin{4};
    Input_u = varargin{5};
    Input_k = varargin{6};
end
try
    swd = xSPM.swd;
    sts = 1;
catch
    sts = 1;
    spmmatfile = spm_mat_path;
    swd = spm_str_manip(spmmatfile,'H');
end
if ~sts, SPM = []; xSPM = []; return; end

%-Preliminaries...
%==========================================================================

%-Load SPM.mat
%--------------------------------------------------------------------------
try
    load(fullfile(swd,'SPM.mat'));
catch
    error(['Cannot read ' fullfile(swd,'SPM.mat')]);
end
SPM.swd = swd;


%-Change directory so that relative filenames are valid
%--------------------------------------------------------------------------
cd(SPM.swd);

%-Check the model has been estimated
%--------------------------------------------------------------------------
try
    SPM.xVol.S;
catch
    
    %-Check the model has been estimated
    %----------------------------------------------------------------------
    str = { 'This model has not been estimated.';...
            'Would you like to estimate it now?'};
    if spm_input(str,1,'bd','yes|no',[1,0],1)
        SPM = spm_spm(SPM);
    else
        SPM = []; xSPM = [];
        return
    end
end

xX   = SPM.xX;                      %-Design definition structure
XYZ  = SPM.xVol.XYZ;                %-XYZ coordinates
S    = SPM.xVol.S;                  %-search Volume {voxels}
R    = SPM.xVol.R;                  %-search Volume {resels}
M    = SPM.xVol.M(1:3,1:3);         %-voxels to mm matrix
VOX  = sqrt(diag(M'*M))';           %-voxel dimensions

%==========================================================================
% - C O N T R A S T S ,   S P M    C O M P U T A T I O N ,    M A S K I N G
%==========================================================================

%-Get contrasts
%--------------------------------------------------------------------------
try, xCon = SPM.xCon; catch, xCon = {}; end

try
    Ic        = xSPM.Ic;
catch
    Ic = InputIC;
end
if isempty(xCon)
    % figure out whether new contrasts were defined, but not selected
    % do this by comparing length of SPM.xCon to xCon, remember added
    % indices to run spm_contrasts on them as well
    try
        noxCon = numel(SPM.xCon);
    catch
        noxCon = 0;
    end
    IcAdd = (noxCon+1):numel(xCon);
else
    IcAdd = [];
end

nc        = length(Ic);  % Number of contrasts

%-Allow user to extend the null hypothesis for conjunctions
%
% n: conjunction number
% u: Null hyp is k<=u effects real; Alt hyp is k>u effects real
%    (NB Here u is from Friston et al 2004 paper, not statistic thresh).
%                  u         n
% Conjunction Null nc-1      1     |    u = nc-n
% Intermediate     1..nc-2   nc-u  |    #effects under null <= u
% Global Null      0         nc    |    #effects under alt  > u,  >= u+1
%----------------------------------+---------------------------------------
if nc > 1
    try
        n = xSPM.n;
    catch
        if nc==2
            But = 'Conjunction|Global';      Val=[1 nc];
        else
            But = 'Conj''n|Intermed|Global'; Val=[1 NaN nc];
        end
        n = spm_input('Null hyp. to assess?','+1','b',But,Val,1);
        if isnan(n)
            if nc == 3,
                n = nc - 1;
            else
                n = nc - spm_input('Effects under null ','0','n1','1',nc-1);
            end
        end
    end
else
    n = 1;
end

%-Enforce orthogonality of multiple contrasts for conjunction
% (Orthogonality within subspace spanned by contrasts)
%--------------------------------------------------------------------------
if nc > 1 && n > 1 && ~spm_FcUtil('|_?',xCon(Ic), xX.xKXs)
    
    OrthWarn = 0;
    
    %-Successively orthogonalise
    %-NB: This loop is peculiarly controlled to account for the
    %     possibility that Ic may shrink if some contrasts disappear
    %     on orthogonalisation (i.e. if there are colinearities)
    %----------------------------------------------------------------------
    i = 1;
    while(i < nc), i = i + 1;
        
        %-Orthogonalise (subspace spanned by) contrast i w.r.t. previous
        %------------------------------------------------------------------
        oxCon = spm_FcUtil('|_',xCon(Ic(i)), xX.xKXs, xCon(Ic(1:i-1)));
        
        %-See if this orthogonalised contrast has already been entered
        % or is colinear with a previous one. Define a new contrast if
        % neither is the case.
        %------------------------------------------------------------------
        d     = spm_FcUtil('In',oxCon,xX.xKXs,xCon);
        
        if spm_FcUtil('0|[]',oxCon,xX.xKXs)
            
            %-Contrast was colinear with a previous one - drop it
            %--------------------------------------------------------------
            Ic(i) = [];
            i     = i - 1;
            
        elseif any(d)
            
            %-Contrast unchanged or already defined - note index
            %--------------------------------------------------------------
            Ic(i) = min(d);
            
        else
            
            %-Define orthogonalised contrast as new contrast
            %--------------------------------------------------------------
            OrthWarn   = OrthWarn + 1;
            conlst     = sprintf('%d,',Ic(1:i-1));
            oxCon.name = sprintf('%s (orth. w.r.t {%s})', xCon(Ic(i)).name,...
                                  conlst(1:end-1));
            xCon       = [xCon, oxCon];
            Ic(i)      = length(xCon);
        end
        
    end % while...
    
    if OrthWarn
        warning('SPM:ConChange','%d contrasts orthogonalized',OrthWarn)
    end
    
    SPM.xCon = xCon;
end % if nc>1...
SPM.xCon = xCon;

%-Apply masking
%--------------------------------------------------------------------------
try
    Mask = ~isempty(xSPM.Im) * (isnumeric(xSPM.Im) + 2*iscellstr(xSPM.Im));
catch
    % Mask = spm_input('mask with other contrast(s)','+1','y/n',[1,0],2);
    Mask = InputMask;  %  0 none  1 contrast  2 image =================================================
end
if Mask == 1
    
    %-Get contrasts for masking
    %----------------------------------------------------------------------
    try
        Im = xSPM.Im;
    catch
        [Im,xCon] = spm_conman(SPM,'T&F',-Inf,...
            'Select contrasts for masking...',' for masking',1);
    end
    
    %-Threshold for mask (uncorrected p-value)
    %----------------------------------------------------------------------
    try
        pm = xSPM.pm;
    catch
        pm = spm_input('uncorrected mask p-value','+1','r',0.05,1,[0,1]);
    end
    
    %-Inclusive or exclusive masking
    %----------------------------------------------------------------------
    try
        Ex = xSPM.Ex;
    catch
        Ex = spm_input('nature of mask','+1','b','inclusive|exclusive',[0,1],1);
    end
    
elseif Mask == 2
    
    %-Get mask images
    %----------------------------------------------------------------------
    try
        Im = xSPM.Im;
    catch
        Im = cellstr(spm_select([1 Inf],'image','Select mask image(s)'));
    end
    
    %-Inclusive or exclusive masking
    %----------------------------------------------------------------------
    try
        Ex = xSPM.Ex;
    catch
        Ex = spm_input('nature of mask','+1','b','inclusive|exclusive',[0,1],1);
    end
    
    pm = [];
    
else
    Im = [];
    pm = [];
    Ex = [];
end


%-Create/Get title string for comparison
%--------------------------------------------------------------------------
if nc == 1
    str  = xCon(Ic).name;
else
    str  = [sprintf('contrasts {%d',Ic(1)),sprintf(',%d',Ic(2:end)),'}'];
    if n == nc
        str = [str ' (global null)'];
    elseif n == 1
        str = [str ' (conj. null)'];
    else
        str = [str sprintf(' (Ha: k>=%d)',(nc-n)+1)];
    end
end
if Ex
    mstr = 'masked [excl.] by';
else
    mstr = 'masked [incl.] by';
end
if isnumeric(Im)
    if length(Im) == 1
        str = sprintf('%s (%s %s at p=%g)',str,mstr,xCon(Im).name,pm);
    elseif ~isempty(Im)
        str = [sprintf('%s (%s {%d',str,mstr,Im(1)),...
            sprintf(',%d',Im(2:end)),...
            sprintf('} at p=%g)',pm)];
    end
elseif iscellstr(Im) && numel(Im) > 0
    [pf,nf,ef] = spm_fileparts(Im{1});
    str  = sprintf('%s (%s %s',str,mstr,[nf ef]);
    for i=2:numel(Im)
        [pf,nf,ef] = spm_fileparts(Im{i});
        str =[str sprintf(', %s',[nf ef])];
    end
    str = [str ')'];
end
try
    titlestr = xSPM.title;
    if isempty(titlestr)
        titlestr = str;
    end
catch
    titlestr = str;% 名称，可以自定义========================================================================
end


%-Bayesian or classical Inference?
%==========================================================================
if isfield(SPM,'PPM')
    
    % Make sure SPM.PPM.xCon field exists
    %----------------------------------------------------------------------
    if ~isfield(SPM.PPM,'xCon')
        SPM.PPM.xCon = [];
    end
    
    % Set Bayesian con type
    %----------------------------------------------------------------------
    SPM.PPM.xCon(Ic).PSTAT = xCon(Ic).STAT;
    
    % Make all new contrasts Bayesian contrasts 
    %----------------------------------------------------------------------
    [xCon(Ic).STAT] = deal('P');
    
    if all(strcmp([SPM.PPM.xCon(Ic).PSTAT],'T'))
        
        % Simple contrast
        %------------------------------------------------------------------
        str = 'Effect size threshold for PPM';
        
        if isfield(SPM.PPM,'VB') % 1st level Bayes
            
            % For VB - set default effect size to zero
            %--------------------------------------------------------------
            Gamma = 0;
            xCon(Ic).eidf = spm_input(str,'+1','e',sprintf('%0.2f',Gamma));
            
        elseif nc == 1 && isempty(xCon(Ic).Vcon) % 2nd level Bayes
            % con image not yet written
            %--------------------------------------------------------------
            if spm_input('Inference',1,'b',{'Bayesian','classical'},[1 0]);
                
                %-Get Bayesian threshold (Gamma) stored in xCon(Ic).eidf
                % The default is one conditional s.d. of the contrast
                %----------------------------------------------------------
                Gamma         = sqrt(xCon(Ic).c'*SPM.PPM.Cb*xCon(Ic).c);
                xCon(Ic).eidf = spm_input(str,'+1','e',sprintf('%0.2f',Gamma));
                xCon(Ic).STAT = 'P';
            end
        end
    else
        % Compound contrast using Chi^2 statistic
        %------------------------------------------------------------------
        if ~isfield(xCon(Ic),'eidf') || isempty(xCon(Ic).eidf)
            xCon(Ic).eidf = 0; % temporarily
        end
    end
end


%-Compute & store contrast parameters, contrast/ESS images, & SPM images
%==========================================================================
SPM.xCon = xCon;
if isnumeric(Im)
    SPM  = spm_contrasts(SPM, unique([Ic, Im, IcAdd]));
else
    SPM  = spm_contrasts(SPM, unique([Ic, IcAdd]));
end
xCon     = SPM.xCon;
STAT     = xCon(Ic(1)).STAT;
VspmSv   = cat(1,xCon(Ic).Vspm);

%-Check conjunctions - Must be same STAT w/ same df
%--------------------------------------------------------------------------
if (nc > 1) && (any(diff(double(cat(1,xCon(Ic).STAT)))) || ...
                any(abs(diff(cat(1,xCon(Ic).eidf))) > 1))
    error('illegal conjunction: can only conjoin SPMs of same STAT & df');
end


%-Degrees of Freedom and STAT string describing marginal distribution
%--------------------------------------------------------------------------
df     = [xCon(Ic(1)).eidf xX.erdf];
if nc > 1
    if n > 1
        str = sprintf('^{%d \\{Ha:k\\geq%d\\}}',nc,(nc-n)+1);
    else
        str = sprintf('^{%d \\{Ha:k=%d\\}}',nc,(nc-n)+1);
    end
else
    str = '';
end

switch STAT
    case 'T'
        STATstr = sprintf('%c%s_{%.0f}','T',str,df(2));
    case 'F'
        STATstr = sprintf('%c%s_{%.0f,%.0f}','F',str,df(1),df(2));
    case 'P'
        STATstr = sprintf('%s^{%0.2f}','PPM',df(1));
end


%-Compute (unfiltered) SPM pointlist for masked conjunction requested
%==========================================================================
fprintf('\t%-32s: %30s','SPM computation','...initialising')            %-#


%-Compute conjunction as minimum of SPMs
%--------------------------------------------------------------------------
Z     = Inf;
for i = Ic
    Z = min(Z,spm_get_data(xCon(i).Vspm,XYZ));
end


%-Copy of Z and XYZ before masking, for later use with FDR
%--------------------------------------------------------------------------
XYZum = XYZ;
Zum   = Z;

%-Compute mask and eliminate masked voxels
%--------------------------------------------------------------------------
for i = 1:numel(Im)
    
    fprintf('%s%30s',repmat(sprintf('\b'),1,30),'...masking')           %-#
    if isnumeric(Im)
        Mask = spm_get_data(xCon(Im(i)).Vspm,XYZ);
        um   = spm_u(pm,[xCon(Im(i)).eidf,xX.erdf],xCon(Im(i)).STAT);
        if Ex
            Q = Mask <= um;
        else
            Q = Mask >  um;
        end
    else
        v = spm_vol(Im{i});
        Mask = spm_get_data(v,v.mat\SPM.xVol.M*[XYZ; ones(1,size(XYZ,2))]);
        Q = Mask ~= 0 & ~isnan(Mask);
        if Ex, Q = ~Q; end
    end
    XYZ   = XYZ(:,Q);
    Z     = Z(Q);
    if isempty(Q)
        fprintf('\n')                                                   %-#
        warning('SPM:NoVoxels','No voxels survive masking at p=%4.2f',pm);
        break
    end
end


%==========================================================================
% - H E I G H T   &   E X T E N T   T H R E S H O L D S
%==========================================================================

u   = -Inf;        % height threshold
k   = 0;           % extent threshold {voxels}

%-Get FDR mode
%--------------------------------------------------------------------------
try
    topoFDR = spm_get_defaults('stats.topoFDR');
catch
    topoFDR = true;
end

%-Height threshold - classical inference
%--------------------------------------------------------------------------
if STAT ~= 'P'
    
    %-Get height threshold
    %----------------------------------------------------------------------
    fprintf('%s%30s',repmat(sprintf('\b'),1,30),'...height threshold')  %-#
    try
        thresDesc = xSPM.thresDesc;
    catch
        if topoFDR
            str = 'FWE|none';
        else
            str = 'FWE|FDR|none';
        end
        thresDesc = InputthresDesc;%===p value adjustment to control: 'FWE' or 'none'====================================================================
    end
    
    switch thresDesc
        
        case 'FWE' % Family-wise false positive rate
            %--------------------------------------------------------------
            try
                u = xSPM.u;
            catch
                u = spm_input('p value (FWE)','+0','r',0.05,1,[0,1]);
            end
            thresDescDes = ['p<' num2str(u) ' (' thresDesc ')'];
            u = spm_uc(u,df,STAT,R,n,S);
            
            
        case 'FDR' % False discovery rate
            %--------------------------------------------------------------
            if topoFDR,
                fprintf('\n');                                          %-#
                error('Change defaults.stats.topoFDR to use voxel FDR');
            end
            try
                u = xSPM.u;
            catch
                u = spm_input('p value (FDR)','+0','r',0.05,1,[0,1]);
            end
            thresDescDes = ['p<' num2str(u) ' (' thresDesc ')'];
            u = spm_uc_FDR(u,df,STAT,n,VspmSv,0);
            
        case 'none'  % No adjustment: p for conjunctions is p of the conjunction SPM
            %--------------------------------------------------------------
            try
                u = xSPM.u;
            catch
                u = Input_u; %threshold  默认0.001========================================================================
            end
            if u <= 1
                thresDescDes = ['p<' num2str(u) ' (unc.)'];
                u = spm_u(u^(1/n),df,STAT);
            else
                thresDescDes = [STAT '=' num2str(u) ];
            end
            
            
        otherwise
            %--------------------------------------------------------------
            fprintf('\n');                                              %-#
            error('Unknown control method "%s".',thresDesc);
            
    end % switch thresDesc
    
    %-Compute p-values for topological and voxel-wise FDR (all search voxels)
    %----------------------------------------------------------------------
    if ~topoFDR
        fprintf('%s%30s',repmat(sprintf('\b'),1,30),'...for voxelFDR')  %-#
        switch STAT
            case 'Z'
                Ps = (1-spm_Ncdf(Zum)).^n;
            case 'T'
                Ps = (1 - spm_Tcdf(Zum,df(2))).^n;
            case 'X'
                Ps = (1-spm_Xcdf(Zum,df(2))).^n;
            case 'F'
                Ps = (1 - spm_Fcdf(Zum,df)).^n;
        end
        Ps = sort(Ps);
    end
    
    %-Peak FDR
    %----------------------------------------------------------------------
    [up,Pp] = spm_uc_peakFDR(0.05,df,STAT,R,n,Zum,XYZum,u);
    
    %-Cluster FDR
    %----------------------------------------------------------------------
    if STAT == 'T' && n == 1
        V2R        = 1/prod(SPM.xVol.FWHM(SPM.xVol.DIM > 1));
        [uc,Pc,ue] = spm_uc_clusterFDR(0.05,df,STAT,R,n,Zum,XYZum,V2R,u);
    else
        uc  = NaN;
        ue  = NaN;
        Pc  = [];
    end
    
    %-Peak FWE
    %----------------------------------------------------------------------
    uu      = spm_uc(0.05,df,STAT,R,n,S);
    
    
%-Height threshold - Bayesian inference
%--------------------------------------------------------------------------
elseif STAT == 'P'
    
    u_default = 1 - 1/SPM.xVol.S;
    str       = 'Posterior probability threshold for PPM';
    u         = spm_input(str,'+0','r',u_default,1);
    thresDescDes = ['P>'  num2str(u) ' (PPM)'];
    
end % (if STAT)

%-Calculate height threshold filtering
%--------------------------------------------------------------------------
Q      = find(Z > u);

%-Apply height threshold
%--------------------------------------------------------------------------
Z      = Z(:,Q);
XYZ    = XYZ(:,Q);
if isempty(Q)
    fprintf('\n');                                                      %-#
    warning('SPM:NoVoxels','No voxels survive height threshold at u=%0.2g',u);
end


%-Extent threshold
%--------------------------------------------------------------------------
if ~isempty(XYZ)
    
    fprintf('%s%30s',repmat(sprintf('\b'),1,30),'...extent threshold'); %-#
    
    %-Get extent threshold [default = 0]
    %----------------------------------------------------------------------
    try
        k = xSPM.k;
    catch
        k = Input_k;
    end
    
    %-Calculate extent threshold filtering
    %----------------------------------------------------------------------
    A     = spm_clusters(XYZ);
    Q     = [];
    for i = 1:max(A)
        j = find(A == i);
        if length(j) >= k, Q = [Q j]; end
    end
    
    % ...eliminate voxels
    %----------------------------------------------------------------------
    Z     = Z(:,Q);
    XYZ   = XYZ(:,Q);
    if isempty(Q)
        fprintf('\n');                                                  %-#
        warning('SPM:NoVoxels','No voxels survive extent threshold at k=%0.2g',k);
    end
    
else
    
    k = 0;
    
end % (if ~isempty(XYZ))

%-For Bayesian inference provide (default) option to display contrast values
%--------------------------------------------------------------------------
if STAT == 'P'
    if spm_input('Plot effect-size/statistic',1,'b',{'Yes','No'},[1 0])
        Z = spm_get_data(xCon(Ic).Vcon,XYZ);
    end
end

%==========================================================================
% - E N D
%==========================================================================
fprintf('%s%30s\n',repmat(sprintf('\b'),1,30),'...done')                %-#
spm('Pointer','Arrow')

%-Assemble output structures of unfiltered data
%==========================================================================
xSPM   = struct( ...
            'swd',      swd,...
            'title',    titlestr,...
            'Z',        Z,...
            'n',        n,...
            'STAT',     STAT,...
            'df',       df,...
            'STATstr',  STATstr,...
            'Ic',       Ic,...
            'Im',       {Im},...
            'pm',       pm,...
            'Ex',       Ex,...
            'u',        u,...
            'k',        k,...
            'XYZ',      XYZ,...
            'XYZmm',    SPM.xVol.M(1:3,:)*[XYZ; ones(1,size(XYZ,2))],...
            'S',        SPM.xVol.S,...
            'R',        SPM.xVol.R,...
            'FWHM',     SPM.xVol.FWHM,...
            'M',        SPM.xVol.M,...
            'iM',       SPM.xVol.iM,...
            'DIM',      SPM.xVol.DIM,...
            'VOX',      VOX,...
            'Vspm',     VspmSv,...
            'thresDesc',thresDesc);

%-RESELS per voxel (density) if it exists
%--------------------------------------------------------------------------
try, xSPM.VRpv = SPM.xVol.VRpv; end
try
    xSPM.units = SPM.xVol.units;
catch
    try, xSPM.units = varargin{1}.units; end
end

%-p-values for topological and voxel-wise FDR
%--------------------------------------------------------------------------
try, xSPM.Ps    = Ps;             end  % voxel   FDR
try, xSPM.Pp    = Pp;             end  % peak    FDR
try, xSPM.Pc    = Pc;             end  % cluster FDR

%-0.05 critical thresholds for FWEp, FDRp, FWEc, FDRc
%--------------------------------------------------------------------------
try, xSPM.uc    = [uu up ue uc];  end
```

## spm_regions_extend.m
```matlab
function [Y,xY] = spm_regions_extend(xSPM,SPM,hReg,xY,Input_VOINames,Input_is,Input_VOI_path)
% VOI time-series extraction of adjusted data (& local eigenimage analysis)
% FORMAT [Y xY] = spm_regions(xSPM,SPM,hReg,[xY]);

% FORMAT [Y xY] = spm_regions(xSPM,SPM,hReg,[xY],Input_VOIName,Input_i)
% Input_VOIName : 抽取的VOI的名称
% Input_i : adjust_contrst 这里选择那个F-All的,整数，1 是dont adjust, 2 是F-All.
% Input_VOI_path : 抽取VOI所使用的Mask的路径;

if nargin < 4, xY = []; end

if nargin > 5
    Input_VOIName = Input_VOINames;
    Input_i =Input_is;
    xY = [];
    
end
%-Get figure handles
%--------------------------------------------------------------------------
Finter = spm_figure('FindWin','Interactive');
if isempty(Finter), noGraph = 1; else noGraph = 0; end
header = get(Finter,'Name');
set(Finter,'Name','VOI time-series extraction');
if ~noGraph, Fgraph = spm_figure('GetWin','Graphics'); end

%-Find nearest voxel [Euclidean distance] in point list
%--------------------------------------------------------------------------
if isempty(xSPM.XYZmm)
    spm('alert!','No suprathreshold voxels!',mfilename,0);
    Y = []; xY = [];
    return
end
try
    xyz    = xY.xyz;
catch
    xyz    = spm_XYZreg('NearestXYZ',...
             spm_XYZreg('GetCoords',hReg),xSPM.XYZmm);
    xY.xyz = xyz;
end

% and update GUI location
%--------------------------------------------------------------------------
spm_XYZreg('SetCoords',xyz,hReg);


%-Get adjustment options and VOI name
%--------------------------------------------------------------------------
if ~noGraph
    if ~isempty(xY.xyz)
        posstr = sprintf('at [%3.0f %3.0f %3.0f]',xY.xyz);
    else
        posstr = '';
    end
    spm_input(posstr,1,'d','VOI time-series extraction');
end

if ~isfield(xY,'name')
    xY.name    = Input_VOIName; %name of region char字符串类型 ========================================================================
end

if ~isfield(xY,'Ic')
    q     = 0;
    Con   = {'<don''t adjust>'};
    for i = 1:length(SPM.xCon)
        if strcmp(SPM.xCon(i).STAT,'F')
            q(end + 1) = i;
            Con{end + 1} = SPM.xCon(i).name;
        end
    end
    i     = Input_i; % adjust_contrst 这里选择那个F-All的,整数，1 是dont adjust, 2 是F-All. =====================================================================
    xY.Ic = q(i);
end

%-If fMRI data then ask user to select session
%--------------------------------------------------------------------------
if isfield(SPM,'Sess') && ~isfield(xY,'Sess')
    s       = length(SPM.Sess);
    if s > 1
        s   = spm_input('which session','!+1','n1',s,s);
    end
    xY.Sess = s;
end

%-Specify VOI
%--------------------------------------------------------------------------
xY.M = xSPM.M;
[xY, xY.XYZmm, Q] = spm_ROI_extend(xY, xSPM.XYZmm,Input_VOI_path);%--------------------------------------------------------------------
try, xY = rmfield(xY,'M'); end
try, xY = rmfield(xY,'rej'); end

if isempty(xY.XYZmm)
    warning('Empty region.');
    Y = [];
    return;
end


%-Extract required data from results files
%==========================================================================
spm('Pointer','Watch')

%-Get raw data, whiten and filter 
%--------------------------------------------------------------------------
y        = spm_get_data(SPM.xY.VY,xSPM.XYZ(:,Q));
y        = spm_filter(SPM.xX.K,SPM.xX.W*y);


%-Computation
%==========================================================================

%-Remove null space of contrast
%--------------------------------------------------------------------------
if xY.Ic

    %-Parameter estimates: beta = xX.pKX*xX.K*y
    %----------------------------------------------------------------------
    beta  = spm_get_data(SPM.Vbeta,xSPM.XYZ(:,Q));

    %-subtract Y0 = XO*beta,  Y = Yc + Y0 + e
    %----------------------------------------------------------------------
    y     = y - spm_FcUtil('Y0',SPM.xCon(xY.Ic),SPM.xX.xKXs,beta);

end

%-Confounds
%--------------------------------------------------------------------------
xY.X0     = SPM.xX.xKXs.X(:,[SPM.xX.iB SPM.xX.iG]);

%-Extract session-specific rows from data and confounds
%--------------------------------------------------------------------------
try
    i     = SPM.Sess(xY.Sess).row;
    y     = y(i,:);
    xY.X0 = xY.X0(i,:);
end

% and add session-specific filter confounds
%--------------------------------------------------------------------------
try
    xY.X0 = [xY.X0 SPM.xX.K(xY.Sess).X0];
end
try
    xY.X0 = [xY.X0 SPM.xX.K(xY.Sess).KH]; % Compatibility check
end

%-Remove null space of X0
%--------------------------------------------------------------------------
xY.X0     = xY.X0(:,any(xY.X0));


%-Compute regional response in terms of first eigenvariate
%--------------------------------------------------------------------------
[m n]   = size(y);
if m > n
    [v s v] = svd(y'*y);
    s       = diag(s);
    v       = v(:,1);
    u       = y*v/sqrt(s(1));
else
    [u s u] = svd(y*y');
    s       = diag(s);
    u       = u(:,1);
    v       = y'*u/sqrt(s(1));
end
d       = sign(sum(v));
u       = u*d;
v       = v*d;
Y       = u*sqrt(s(1)/n);

%-Set in structure
%--------------------------------------------------------------------------
xY.y    = y;
xY.u    = Y;
xY.v    = v;
xY.s    = s;

%-Display VOI weighting and eigenvariate
%==========================================================================
if ~noGraph
    
    % show position
    %----------------------------------------------------------------------
    spm_results_ui('Clear',Fgraph);
    figure(Fgraph);
    subplot(2,2,3)
    spm_dcm_display(xY)

    % show dynamics
    %----------------------------------------------------------------------
    subplot(2,2,4)
    try
        plot(SPM.xY.RT*[1:length(xY.u)],Y)
        str = 'time (seconds}';
    catch
        plot(Y)
        str = 'scan';
    end
    title(['1st eigenvariate: ' xY.name],'FontSize',10)
    if strcmpi(xY.def,'mask')
        [p,n,e] = fileparts(xY.spec.fname);
        posstr  = sprintf('from mask %s', [n e]);
    else
        posstr  = sprintf('at [%3.0f %3.0f %3.0f]',xY.xyz);
    end
    str = { str;' ';...
        sprintf('%d voxels in VOI %s',length(Q),posstr);...
        sprintf('Variance: %0.2f%%',s(1)*100/sum(s))};
    xlabel(str)
    axis tight square
end

%-Save
%==========================================================================
str = ['VOI_' xY.name '.mat'];
if isfield(xY,'Sess') && isfield(SPM,'Sess')
    str = sprintf('VOI_%s_%i.mat',xY.name,xY.Sess);
end
if spm_check_version('matlab','7') >= 0
    save(fullfile(SPM.swd,str),'-V6','Y','xY')
else
    save(fullfile(SPM.swd,str),'Y','xY')
end

fprintf('   VOI saved as %s\n',spm_str_manip(fullfile(SPM.swd,str),'k55'));

%-Reset title
%--------------------------------------------------------------------------
set(Finter,'Name',header);
spm('Pointer','Arrow')
```

## spm_result_ui_extend.m
```matlab
function varargout = spm_result_ui_extend(varargin)

% FORMAT [hreg,xSPM,SPM] = spm_results_ui('Setup')
% Query SPM and setup GUI. 
%
% FORMAT [hreg,xSPM,SPM] = spm_results_ui('Setup',xSPM)
% Query SPM and setup GUI using a xSPM input structure. This allows to run
% results setup without user interaction. See spm_getSPM for details of
% allowed fields.
%
% FORMAT hReg = spm_results_ui('SetupGUI',M,DIM,xSPM,Finter)
% Setup results GUI in Interactive window
% M       - 4x4 transformation matrix relating voxel to "real" co-ordinates
% DIM     - 3 vector of image X, Y & Z dimensions
% xSPM    - structure containing xSPM. Required fields are:
% .Z      - minimum of n Statistics {filtered on u and k}
% .XYZmm  - location of voxels {mm}
% Finter  - handle (or 'Tag') of Interactive window (default 'Interactive')
% hReg    - handle of XYZ registry object
%
% FORMAT spm_results_ui('DrawButts',hReg,DIM,Finter,WS,FS)
% Draw GUI buttons
% hReg    - handle of XYZ registry object
% DIM     - 3 vector of image X, Y & Z dimensions
% Finter  - handle of Interactive window
% WS      - WinScale  [Default spm('WinScale') ]
% FS      - FontSizes [Default spm('FontSizes')]
%
% FORMAT hFxyz = spm_results_ui('DrawXYZgui',M,DIM,xSPM,xyz,Finter)
% Setup editable XYZ control widgets at foot of Interactive window
% M      - 4x4 transformation matrix relating voxel to "real" co-ordinates
% DIM    - 3 vector of image X, Y & Z dimensions
% xSPM   - structure containing SPM; Required fields are:
% .Z     - minimum of n Statistics {filtered on u and k}
% .XYZmm - location of voxels {mm}
% xyz    - Initial xyz location {mm}
% Finter - handle of Interactive window
% hFxyz  - handle of XYZ control - the frame containing the edit widgets
%
% FORMAT spm_results_ui('EdWidCB')
% Callback for editable XYZ control widgets
%
% FORMAT spm_results_ui('UpdateSPMval',hFxyz)
% FORMAT spm_results_ui('UpdateSPMval',UD)
% Updates SPM value string in Results GUI (using data from UserData of hFxyz)
% hFxyz - handle of frame enclosing widgets - the Tag object for this control
% UD    - XYZ data structure (UserData of hFxyz).
%
% FORMAT xyz = spm_results_ui('GetCoords',hFxyz)
% Get current co-ordinates from editable XYZ control
% hFxyz - handle of frame enclosing widgets - the Tag object for this control
% xyz   - current co-ordinates {mm}
% NB: When using the results section, should use XYZregistry to get/set location
%
% FORMAT [xyz,d] = spm_results_ui('SetCoords',xyz,hFxyz,hC)
% Set co-ordinates to XYZ widget
% xyz   - (Input) desired co-ordinates {mm}
% hFxyz - handle of XYZ control - the frame containing the edit widgets
% hC    - handle of calling object, if used as a callback. [Default 0]
% xyz   - (Output) Desired co-ordinates are rounded to nearest voxel if hC
%         is not specified, or is zero. Otherwise, caller is assumed to
%         have checked verity of desired xyz co-ordinates. Output xyz returns
%         co-ordinates actually set {mm}.
% d     - Euclidean distance between desired and set co-ordinates.
% NB: When using the results section, should use XYZregistry to get/set location
%
% FORMAT hFxyz = spm_results_ui('FindXYZframe',h)
% Find/check XYZ edit widgets frame handle, 'Tag'ged 'hFxyz'
% h     - handle of frame enclosing widgets, or containing figure [default gcf]
%         If ischar(h), then uses spm_figure('FindWin',h) to locate named figures
% hFxyz - handle of confirmed XYZ editable widgets control
%         Errors if hFxyz is not an XYZ widget control, or a figure containing
%         a unique such control
%
% FORMAT spm_results_ui('PlotUi',hAx)
% GUI for adjusting plot attributes - Sets up controls just above results GUI
% hAx - handle of axes to work with
%
% FORMAT spm_results_ui('PlotUiCB')
% CallBack handler for Plot attribute GUI
%
% FORMAT Fgraph = spm_results_ui('Clear',F,mode)
% Clears results subpane of Graphics window, deleting all but semi-permanent
% results section stuff
% F      - handle of Graphics window [Default spm_figure('FindWin','Graphics')]
% mode   - 1 [default] - clear results subpane
%        - 0           - clear results subpane and hide results stuff
%        - 2           - clear, but respect 'NextPlot' 'add' axes
%                        (which is set by `hold on`)
% Fgraph - handle of Graphics window
%
% FORMAT hMP = spm_results_ui('LaunchMP',M,DIM,hReg,hBmp)
% Prototype callback handler for integrating MultiPlanar toolbox
%
% FORMAT spm_results_ui('Delete',h)
% deletes HandleGraphics objects, but only if they're valid, thus avoiding
% warning statements from MATLAB.
%__________________________________________________________________________
 
SVNid = '$Rev: 4209 $'; 

%-Condition arguments
%--------------------------------------------------------------------------
if nargin == 0, Action='SetUp'; else Action=varargin{1}; end
 
 
%==========================================================================
switch lower(Action), case 'setup'                         %-Set up results
%==========================================================================
 
    %-Initialise
    %----------------------------------------------------------------------
    SPMid = spm('FnBanner',mfilename,SVNid);
    [Finter,Fgraph,CmdLine] = spm('FnUIsetup','Stats: Results');
    spm_clf('Satellite')
    FS    = spm('FontSizes');
 
    %-Get thresholded xSPM data and parameters of design
    %======================================================================
    if nargin > 1
        [SPM,xSPM] = spm_getSPM_extend(varargin{2});
    else
        [SPM,xSPM] = spm_getSPM;
    end
 
    if isempty(xSPM) 
        varargout = {[],[],[]};
        return;
    end
 
    %-Ensure pwd = swd so that relative filenames are valid
    %----------------------------------------------------------------------
    cd(SPM.swd)
    
    %-Get space information
    %======================================================================
    M         = SPM.xVol.M;
    DIM       = SPM.xVol.DIM;

    %-Space units
    %----------------------------------------------------------------------
    try
        try
            units = SPM.xVol.units;
        catch
            units = xSPM.units;
        end
    catch
        try
            if strcmp(spm('CheckModality'),'EEG')
                datatype = {...
                    'Volumetric (2D/3D)',...
                    'Scalp-Time',...
                    'Scalp-Frequency',...
                    'Time-Frequency',...
                    'Frequency-Frequency'};
                selected = spm_input('Data Type: ','+1','m',datatype);
                datatype = datatype{selected};
            else
                datatype = 'Volumetric (2D/3D)';
            end
        catch
            datatype     = 'Volumetric (2D/3D)';
        end
        
        switch datatype
            case 'Volumetric (2D/3D)'
                units    = {'mm' 'mm' 'mm'};
            case 'Scalp-Time'
                units    = {'mm' 'mm' 'ms'};
            case 'Scalp-Frequency'
                units    = {'mm' 'mm' 'Hz'};
            case 'Time-Frequency'
                units    = {'Hz' 'ms' ''};
            case 'Frequency-Frequency'
                units    = {'Hz' 'Hz' ''};
            otherwise
                error('Unknown data type.');
        end
    end
    if DIM(3) == 1, units{3} = ''; end
    xSPM.units      = units;
    SPM.xVol.units  = units;
    
     
    %-Setup Results User Interface; Display MIP, design matrix & parameters
    %======================================================================
 
    %-Setup results GUI
    %----------------------------------------------------------------------
    spm_clf(Finter);
    spm('FigName',['SPM{',xSPM.STAT,'}: Results'],Finter,CmdLine);
    hReg      = spm_results_ui('SetupGUI',M,DIM,xSPM,Finter);
 
    %-Setup design interrogation menu
    %----------------------------------------------------------------------
    hDesRepUI = spm_DesRep('DesRepUI',SPM);
    figure(Finter)
 
    %-Setup contrast menu
    %----------------------------------------------------------------------
    hC = uimenu(Finter,'Label','Contrasts', 'Tag','ContrastsUI');
    hC1 = uimenu(hC,'Label','New Contrast...',...
        'UserData',struct('Ic',0),...
        'Callback',{@mychgcon,xSPM});
    hC1 = uimenu(hC,'Label','Change Contrast');
    for i=1:numel(SPM.xCon)
        hC2 = uimenu(hC1,'Label',[SPM.xCon(i).STAT, ': ', SPM.xCon(i).name], ...
            'UserData',struct('Ic',i),...
            'Callback',{@mychgcon,xSPM});
        if any(xSPM.Ic == i)
            set(hC2,'ForegroundColor',[0 0 1],'Checked','on');
        end
    end
    hC1 = uimenu(hC,'Label','Previous Contrast',...
        'Accelerator','P',...
        'UserData',struct('Ic',xSPM.Ic-1),...
        'Callback',{@mychgcon,xSPM});
    if xSPM.Ic-1<1, set(hC1,'Enable','off'); end
    hC1 = uimenu(hC,'Label','Next Contrast',...
        'Accelerator','N',...
        'UserData',struct('Ic',xSPM.Ic+1),...
        'Callback',{@mychgcon,xSPM});
    if xSPM.Ic+1>numel(SPM.xCon), set(hC1,'Enable','off'); end
    hC1 = uimenu(hC,'Label','Significance level','Separator','on');
    xSPMtmp = xSPM; xSPMtmp.thresDesc = '';
    uimenu(hC1,'Label','Change...','UserData',struct('Ic',xSPM.Ic),...
        'Callback',{@mychgcon,xSPMtmp});
    xSPMtmp = xSPM; xSPMtmp.thresDesc = 'p<0.05 (FWE)';
    uimenu(hC1,'Label','Set to 0.05 (FWE)','UserData',struct('Ic',xSPM.Ic),...
        'Callback',{@mychgcon,xSPMtmp});
    xSPMtmp = xSPM; xSPMtmp.thresDesc = 'p<0.001 (unc.)';
    uimenu(hC1,'Label','Set to 0.001 (unc.)','UserData',struct('Ic',xSPM.Ic),...
        'Callback',{@mychgcon,xSPMtmp});
    uimenu(hC1,'Label',[xSPM.thresDesc ', k=' num2str(xSPM.k)],...
        'Enable','off','Separator','on');
    
    %-Setup Maximum intensity projection (MIP) & register
    %----------------------------------------------------------------------
    hMIPax = axes('Parent',Fgraph,'Position',[0.05 0.60 0.55 0.36],'Visible','off');
    hMIPax = spm_mip_ui(xSPM.Z,xSPM.XYZmm,M,DIM,hMIPax,units);
 
    spm_XYZreg('XReg',hReg,hMIPax,'spm_mip_ui');
    if xSPM.STAT == 'P'
        str = xSPM.STATstr;
    else
        str = ['SPM\{',xSPM.STATstr,'\}'];
    end
    text(240,260,str,...
        'Interpreter','TeX',...
        'FontSize',FS(14),'Fontweight','Bold',...
        'Parent',hMIPax)
 
 
    %-Print comparison title
    %----------------------------------------------------------------------
    hTitAx = axes('Parent',Fgraph,...
        'Position',[0.02 0.95 0.96 0.02],...
        'Visible','off');
 
    text(0.5,0,xSPM.title,'Parent',hTitAx,...
        'HorizontalAlignment','center',...
        'VerticalAlignment','baseline',...
        'FontWeight','Bold','FontSize',FS(14))
 
 
    %-Print SPMresults: Results directory & thresholding info
    %----------------------------------------------------------------------
    hResAx = axes('Parent',Fgraph,...
        'Position',[0.05 0.55 0.45 0.05],...
        'DefaultTextVerticalAlignment','baseline',...
        'DefaultTextFontSize',FS(9),...
        'DefaultTextColor',[1,1,1]*.7,...
        'Units','points',...
        'Visible','off');
    AxPos = get(hResAx,'Position'); set(hResAx,'YLim',[0,AxPos(4)])
    h     = text(0,24,'SPMresults:','Parent',hResAx,...
        'FontWeight','Bold','FontSize',FS(14));
    text(get(h,'Extent')*[0;0;1;0],24,spm_str_manip(SPM.swd,'a30'),'Parent',hResAx)
    try
        thresDesc = xSPM.thresDesc;
        text(0,12,sprintf('Height threshold %c = %0.6f  {%s}',xSPM.STAT,xSPM.u,thresDesc),'Parent',hResAx)
    catch
        text(0,12,sprintf('Height threshold %c = %0.6f',xSPM.STAT,xSPM.u),'Parent',hResAx)
    end
    text(0,00,sprintf('Extent threshold k = %0.0f voxels',xSPM.k), 'Parent',hResAx)
 
 
    %-Plot design matrix
    %----------------------------------------------------------------------
    hDesMtx   = axes('Parent',Fgraph,'Position',[0.65 0.55 0.25 0.25]);
    hDesMtxIm = image((SPM.xX.nKX + 1)*32);
    xlabel('Design matrix')
    set(hDesMtxIm,'ButtonDownFcn','spm_DesRep(''SurfDesMtx_CB'')',...
        'UserData',struct(...
        'X',        SPM.xX.xKXs.X,...
        'fnames',   {reshape({SPM.xY.VY.fname},size(SPM.xY.VY))},...
        'Xnames',   {SPM.xX.name}))
 
    %-Plot contrasts
    %----------------------------------------------------------------------
    nPar   = size(SPM.xX.X,2);
    xx     = [repmat([0:nPar-1],2,1);repmat([1:nPar],2,1)];
    nCon   = length(xSPM.Ic);
    xCon   = SPM.xCon;
    if nCon
        dy     = 0.15/max(nCon,2);
        hConAx = axes('Position',[0.65 (0.80 + dy*.1) 0.25 dy*(nCon-.1)],...
            'Tag','ConGrphAx','Visible','off');
        title('contrast(s)')
        htxt   = get(hConAx,'title');
        set(htxt,'Visible','on','HandleVisibility','on')
    end
 
    for ii = nCon:-1:1
        axes('Position',[0.65 (0.80 + dy*(nCon - ii +.1)) 0.25 dy*.9])
        if xCon(xSPM.Ic(ii)).STAT == 'T' && size(xCon(xSPM.Ic(ii)).c,2) == 1
 
            %-Single vector contrast for SPM{t} - bar
            %--------------------------------------------------------------
            yy = [zeros(1,nPar);repmat(xCon(xSPM.Ic(ii)).c',2,1);zeros(1,nPar)];
            h  = patch(xx,yy,[1,1,1]*.5);
            set(gca,'Tag','ConGrphAx',...
                'Box','off','TickDir','out',...
                'XTick',spm_DesRep('ScanTick',nPar,10) - 0.5,'XTickLabel','',...
                'XLim', [0,nPar],...
                'YTick',[-1,0,+1],'YTickLabel','',...
                'YLim',[min(xCon(xSPM.Ic(ii)).c),max(xCon(xSPM.Ic(ii)).c)] +...
                [-1 +1] * max(abs(xCon(xSPM.Ic(ii)).c))/10  )
 
        else
 
            %-F-contrast - image
            %--------------------------------------------------------------
            h = image((xCon(xSPM.Ic(ii)).c'/max(abs(xCon(xSPM.Ic(ii)).c(:)))+1)*32);
            set(gca,'Tag','ConGrphAx',...
                'Box','on','TickDir','out',...
                'XTick',spm_DesRep('ScanTick',nPar,10),'XTickLabel','',...
                'XLim', [0,nPar]+0.5,...
                'YTick',[0:size(SPM.xCon(xSPM.Ic(ii)).c,2)]+0.5,...
                'YTickLabel','',...
                'YLim', [0,size(xCon(xSPM.Ic(ii)).c,2)]+0.5 )
 
        end
        ylabel(num2str(xSPM.Ic(ii)))
        set(h,'ButtonDownFcn','spm_DesRep(''SurfCon_CB'')',...
            'UserData', struct( 'i',        xSPM.Ic(ii),...
            'h',        htxt,...
            'xCon',     xCon(xSPM.Ic(ii))))
    end
 
 
    %-Store handles of results section Graphics window objects
    %----------------------------------------------------------------------
    H  = get(Fgraph,'Children');
    H  = findobj(H,'flat','HandleVisibility','on');
    H  = findobj(H);
    Hv = get(H,'Visible');
    set(hResAx,'Tag','PermRes','UserData',struct('H',H,'Hv',{Hv}))
 
    %-Finished results setup
    %----------------------------------------------------------------------
    varargout = {hReg,xSPM,SPM};
    spm('Pointer','Arrow')
 
 
    %======================================================================
    case 'setupgui'                            %-Set up results section GUI
    %======================================================================
        % hReg = spm_results_ui('SetupGUI',M,DIM,xSPM,Finter)
        if nargin < 5, Finter='Interactive'; else Finter = varargin{5}; end
        if nargin < 4, error('Insufficient arguments'), end
        M      = varargin{2};
        DIM    = varargin{3};
        Finter = spm_figure('GetWin',Finter);
        WS     = spm('WinScale');
        FS     = spm('FontSizes');
 
        %-Create frame for Results GUI objects
        %------------------------------------------------------------------
        hReg    = uicontrol(Finter,'Style','Frame','Position',[001 001 400 190].*WS,...
                           'BackgroundColor',spm('Colour'));
        hFResUi = uicontrol(Finter,...
                     'Style','Pushbutton',...
                     'enable','off',...
                     'Position',[008 007 387 178].*WS);
 
        %-Initialise registry in hReg frame object
        %------------------------------------------------------------------
        [hReg,xyz] = spm_XYZreg('InitReg',hReg,M,DIM,[0;0;0]);
 
        %-Setup editable XYZ widgets & cross register with registry
        %------------------------------------------------------------------
        hFxyz      = spm_results_ui('DrawXYZgui',M,DIM,varargin{4},xyz,Finter);
        spm_XYZreg('XReg',hReg,hFxyz,'spm_results_ui');
 
        %-Set up buttons for results functions
        %------------------------------------------------------------------
        spm_results_ui('DrawButts',hReg,DIM,Finter,WS,FS);
 
        varargout  = {hReg};
 
 
 
    %======================================================================
    case 'drawbutts'   %-Draw results section buttons in Interactive window
    %======================================================================
        % spm_results_ui('DrawButts',hReg,DIM,Finter,WS,FS)
        %
        if nargin<3, error('Insufficient arguments'), end
        hReg = varargin{2};
        DIM  = varargin{3};
        if nargin<4,  Finter = spm_figure('FindWin','Interactive');
        else Finter = varargin{4}; end
        if nargin < 5, WS = spm('WinScale');  else  WS = varargin{5}; end
        if nargin < 6, FS = spm('FontSizes'); else  FS = varargin{6}; end
 
        %-p-values
        %------------------------------------------------------------------
        uicontrol(Finter,'Style','Text','String','p-values',...
            'Position',[020 168 080 015].*WS,...
            'FontAngle','Italic',...
            'FontSize',FS(10),...
            'HorizontalAlignment','Left',...
            'ForegroundColor','w')
        uicontrol(Finter,'Style','PushButton','String','whole brain','FontSize',FS(10),...
            'ToolTipString',...
            'tabulate summary of local maxima, p-values & statistics',...
            'Callback','spm_list(''List'',xSPM,hReg);',...
            'Interruptible','on','Enable','on',...
            'Position',[015 145 100 020].*WS)
        uicontrol(Finter,'Style','PushButton','String','current cluster','FontSize',FS(10),...
            'ToolTipString',...
            'tabulate p-values & statistics for local maxima of nearest cluster',...
            'Callback','spm_list(''ListCluster'',xSPM,hReg);',...
            'Interruptible','on','Enable','on',...
            'Position',[015 120 100 020].*WS)
        uicontrol(Finter,'Style','PushButton','String','small volume','FontSize',FS(10),...
            'ToolTipString',['Small Volume Correction - corrected p-values ',...
            'for a small search region'],...
            'Callback','spm_VOI(SPM,xSPM,hReg);',...
            'Interruptible','on','Enable','on',...
            'Position',[015 095 100 020].*WS)
 
 
        %-SPM area - used for Volume of Interest analyses
        %------------------------------------------------------------------
        uicontrol(Finter,'Style','Text','String','Multivariate',...
            'Position',[135 168 80 015].*WS,...
            'FontAngle','Italic',...
            'FontSize',FS(10),...
            'HorizontalAlignment','Left',...
            'ForegroundColor','w')
        uicontrol(Finter,'Style','PushButton','String','eigenvariate',...
            'Position',[130 145 70 020].*WS,...
            'ToolTipString',...
            'Responses (principal eigenvariate) in volume of interest',...
            'Callback','[Y,xY] = spm_regions(xSPM,SPM,hReg)',...
            'Interruptible','on','Enable','on',...
            'FontSize',FS(10),'ForegroundColor',[1 1 1]/3)
        uicontrol(Finter,'Style','PushButton','String','CVA',...
            'Position',[205 145 65 020].*WS,...
            'ToolTipString',...
            'Canonical variates analysis for the current contrast and VOI',...
            'Callback','CVA = spm_cva(xSPM,SPM,hReg)',...
            'Interruptible','on','Enable','on',...
            'FontSize',FS(10),'ForegroundColor',[1 1 1]/3)
        uicontrol(Finter,'Style','PushButton','String','multivariate Bayes',...
            'Position',[130 120 140 020].*WS,...
            'ToolTipString',...
            'Multivariate Bayes',...
            'Callback','[MVB] = spm_mvb_ui(xSPM,SPM,hReg)',...
            'Interruptible','on','Enable','on',...
            'FontSize',FS(10),'ForegroundColor',[1 1 1]/3)
        uicontrol(Finter,'Style','PushButton','String','BMS',...
            'Position',[130 95 68 020].*WS,...
            'ToolTipString',...
            'Compare or review a multivariate Bayesian model',...
            'Callback','[F,P]  = spm_mvb_bmc',...
            'Interruptible','on','Enable','on',...
            'FontSize',FS(8),'ForegroundColor',[1 1 1]/3)
        uicontrol(Finter,'Style','PushButton','String','p-value',...
            'Position',[202 95 68 020].*WS,...
            'ToolTipString',...
            'Randomisation testing of a multivariate Bayesian model',...
            'Callback','spm_mvb_p',...
            'Interruptible','on','Enable','on',...
            'FontSize',FS(8),'ForegroundColor',[1 1 1]/3)
 
        %-Hemodynamic modelling
        %------------------------------------------------------------------
        if strcmp(spm('CheckModality'),'FMRI')
            uicontrol(Finter,'Style','PushButton','String','Hemodynamics',...
                'FontSize',FS(10),...
                'ToolTipString','Hemodynamic modelling of regional response',...
                'Callback','[Ep,Cp,K1,K2] = spm_hdm_ui(xSPM,SPM,hReg);',...
                'Interruptible','on','Enable','on',...
                'Position',[130 055 140 020].*WS,...
                'ForegroundColor',[1 1 1]/3);
        end
 
        %-Not currently used
        %------------------------------------------------------------------
        %uicontrol(Finter,'Style','PushButton','String','','FontSize',FS(10),...
        %     'ToolTipString','',...
        %     'Callback','',...
        %     'Interruptible','on','Enable','on',...
        %     'Position',[015 055 100 020].*WS)

        %-Visualisation
        %------------------------------------------------------------------
        uicontrol(Finter,'Style','Text','String','Display',...
            'Position',[290 168 065 015].*WS,...
            'FontAngle','Italic',...
            'FontSize',FS(10),...
            'HorizontalAlignment','Left',...
            'ForegroundColor','w')
        uicontrol(Finter,'Style','PushButton','String','plot','FontSize',FS(10),...
            'ToolTipString','plot data & contrasts at current voxel',...
            'Callback','[Y,y,beta,Bcov] = spm_graph(xSPM,SPM,hReg);',...
            'Interruptible','on','Enable','on',...
            'Position',[285 145 100 020].*WS,...
            'Tag','plotButton')
 
        str  = { 'overlays...','slices','sections','render','previous sections','previous render'};
        tstr = { 'overlay filtered SPM on another image: ',...
            '3 slices / ','ortho sections / ','render /','previous ortho sections /','previous surface rendering'};
 
        tmp  = { 'spm_transverse(''set'',xSPM,hReg)',...
            'spm_sections(xSPM,hReg)',...
            ['spm_render(   struct( ''XYZ'',    xSPM.XYZ,',...
            '''t'',     xSPM.Z'',',...
            '''mat'',   xSPM.M,',...
            '''dim'',   xSPM.DIM))'],...
            ['global prevsect;','spm_sections(xSPM,hReg,prevsect)'],...
            ['global prevrend;','if ~isstruct(prevrend)',...
            'prevrend = struct(''rendfile'','''',''brt'',[],''col'',[]); end;',...            
            'spm_render(    struct( ''XYZ'',    xSPM.XYZ,',...
            '''t'',     xSPM.Z'',',...
            '''mat'',   xSPM.M,',...
            '''dim'',   xSPM.DIM),prevrend.brt,prevrend.rendfile)']};
 
        uicontrol(Finter,'Style','PopUp','String',str,'FontSize',FS(10),...
            'ToolTipString',cat(2,tstr{:}),...
            'Callback','spm(''PopUpCB'',gcbo)',...
            'UserData',tmp,...
            'Interruptible','on','Enable','on',...
            'Position',[285 120 100 020].*WS)
 
        uicontrol(Finter,'Style','PushButton','String','save','FontSize',FS(10),...
            'ToolTipString','save thresholded SPM as image',...
            'Callback',['spm_write_filtered(xSPM.Z,xSPM.XYZ,xSPM.DIM,xSPM.M,',...
              'sprintf(''SPM{%c}-filtered: u = %5.3f, k = %d'',',...
              'xSPM.STAT,xSPM.u,xSPM.k));'],...
            'Interruptible','on','Enable','on',...
            'Position',[285 095 100 020].*WS)
 
        %-ResultsUI controls
        %------------------------------------------------------------------
        hClear = uicontrol(Finter,'Style','PushButton','String','clear',...
            'ToolTipString','clears results subpane',...
            'FontSize',FS(9),'ForegroundColor','b',...
            'Callback',['spm_results_ui(''Clear''); ',...
              'spm_input(''!DeleteInputObj''),',...
              'spm_clf(''Satellite'')'],...
            'Interruptible','on','Enable','on',...
            'DeleteFcn','spm_clf(''Graphics'')',...
            'Position',[285 055 035 018].*WS);
 
        hExit  = uicontrol(Finter,'Style','PushButton','String','exit',...
            'ToolTipString','exit the results section',...
            'FontSize',FS(9),'ForegroundColor','r',...
            'Callback','spm_results_ui(''close'')',...
            'Interruptible','on','Enable','on',...
            'Position',[325 055 035 018].*WS);
 
        hHelp  = uicontrol(Finter,'Style','PushButton','String','?',...
            'ToolTipString','results section help',...
            'FontSize',FS(9),'ForegroundColor','g',...
            'Callback','spm_help(''spm_results_ui'')',...
            'Interruptible','on','Enable','on',...
            'Position',[365 055 020 018].*WS);
 
 
    %======================================================================
    case 'drawxyzgui'                                   %-Draw XYZ GUI area
    %======================================================================
        % hFxyz = spm_results_ui('DrawXYZgui',M,DIM,xSPM,xyz,Finter)
        if nargin<6,  Finter=spm_figure('FindWin','Interactive');
        else Finter=varargin{6}; end
        if nargin < 5, xyz=[0;0;0]; else xyz=varargin{5}; end
        if nargin < 4, error('Insufficient arguments'), end
        DIM     = varargin{3};
        M       = varargin{2};
        xyz     = spm_XYZreg('RoundCoords',xyz,M,DIM);
 
        %-Font details
        %------------------------------------------------------------------
        WS      = spm('WinScale');
        FS      = spm('FontSizes');
        PF      = spm_platform('fonts');
 
        %-Create XYZ control objects
        %------------------------------------------------------------------
        hFxyz = uicontrol(Finter,'Style','Pushbutton',...
            'visible','off','enable','off','Position',[010 010 265 030].*WS);
        uicontrol(Finter,'Style','Text','String','co-ordinates',...
            'Position',[020 035 090 016].*WS,...
            'FontAngle','Italic',...
            'FontSize',FS(10),...
            'HorizontalAlignment','Left',...
            'ForegroundColor','w')
 
        uicontrol(Finter,'Style','Text','String','x =',...
            'Position',[020 015 024 018].*WS,...
            'FontName',PF.times,'FontSize',FS(10),'FontAngle','Italic',...
            'HorizontalAlignment','Center');
        hX   = uicontrol(Finter,'Style','Edit','String',sprintf('%.2f',xyz(1)),...
            'ToolTipString','enter x-coordinate',...
            'Position',[044 015 056 020].*WS,...
            'FontSize',FS(10),'BackGroundColor',[.8,.8,1],...
            'HorizontalAlignment','Right',...
            'Tag','hX',...
            'Callback','spm_results_ui(''EdWidCB'')');
 
        uicontrol(Finter,'Style','Text','String','y =',...
            'Position',[105 015 024 018].*WS,...
            'FontName',PF.times,'FontSize',FS(10),'FontAngle','Italic',...
            'HorizontalAlignment','Center')
        hY   = uicontrol(Finter,'Style','Edit','String',sprintf('%.2f',xyz(2)),...
            'ToolTipString','enter y-coordinate',...
            'Position',[129 015 056 020].*WS,...
            'FontSize',FS(10),'BackGroundColor',[.8,.8,1],...
            'HorizontalAlignment','Right',...
            'Tag','hY',...
            'Callback','spm_results_ui(''EdWidCB'')');
 
        if DIM(3) ~= 1
        uicontrol(Finter,'Style','Text','String','z =',...
            'Position',[190 015 024 018].*WS,...
            'FontName',PF.times,'FontSize',FS(10),'FontAngle','Italic',...
            'HorizontalAlignment','Center')
        hZ   = uicontrol(Finter,'Style','Edit','String',sprintf('%.2f',xyz(3)),...
            'ToolTipString','enter z-coordinate',...
            'Position',[214 015 056 020].*WS,...
            'FontSize',FS(10),'BackGroundColor',[.8,.8,1],...
            'HorizontalAlignment','Right',...
            'Tag','hZ',...
            'Callback','spm_results_ui(''EdWidCB'')');
        else
        hZ = [];
        end
        
        %-Statistic value reporting pane
        %------------------------------------------------------------------
        uicontrol(Finter,'Style','Text','String','statistic',...
            'Position',[285 035 090 016].*WS,...
            'FontAngle','Italic',...
            'FontSize',FS(10),...
            'HorizontalAlignment','Left',...
            'ForegroundColor','w')
        hSPM = uicontrol(Finter,'Style','Text','String','',...
            'Position',[285 012 100 020].*WS,...
            'FontSize',FS(10),...
            'HorizontalAlignment','Center');
 
 
        %-Store data
        %------------------------------------------------------------------
        set(hFxyz,'Tag','hFxyz','UserData',struct(...
            'hReg', [],...
            'M',    M,...
            'DIM',  DIM,...
            'XYZ',  varargin{4}.XYZmm,...
            'Z',    varargin{4}.Z,...
            'hX',   hX,...
            'hY',   hY,...
            'hZ',   hZ,...
            'hSPM', hSPM,...
            'xyz',  xyz ));
 
        set([hX,hY,hZ],'UserData',hFxyz)
        varargout = {hFxyz};
 
 
    %======================================================================
    case 'edwidcb'                          %-Callback for editable widgets
    %======================================================================
        % spm_results_ui('EdWidCB')
 
        hC    = gcbo;
        d     = find(strcmp(get(hC,'Tag'),{'hX','hY','hZ'}));
        hFxyz = get(hC,'UserData');
        UD    = get(hFxyz,'UserData');
        xyz   = UD.xyz;
        nxyz  = xyz;
 
        o = evalin('base',['[',get(hC,'String'),']'],'sprintf(''error'')');
        if ischar(o) || length(o)>1
            warning(sprintf('%s: Error evaluating ordinate:\n\t%s',...
                mfilename,lasterr))
        else
            nxyz(d) = o;
            nxyz = spm_XYZreg('RoundCoords',nxyz,UD.M,UD.DIM);
        end
 
        if abs(xyz(d)-nxyz(d))>0
            UD.xyz = nxyz; set(hFxyz,'UserData',UD)
            if ~isempty(UD.hReg), spm_XYZreg('SetCoords',nxyz,UD.hReg,hFxyz); end
            set(hC,'String',sprintf('%.3f',nxyz(d)))
            spm_results_ui('UpdateSPMval',UD)
        end
 
        
    %======================================================================
    case 'updatespmval'                           %-Update SPM value in GUI
    %======================================================================
        % spm_results_ui('UpdateSPMval',hFxyz)
        % spm_results_ui('UpdateSPMval',UD)
        if nargin<2, error('insufficient arguments'), end
        if isstruct(varargin{2}), UD=varargin{2}; else UD = get(varargin{2},'UserData'); end
        i  = spm_XYZreg('FindXYZ',UD.xyz,UD.XYZ);
        if isempty(i), str = ''; else str = sprintf('%6.2f',UD.Z(i)); end
        set(UD.hSPM,'String',str);
 
 
    %======================================================================
    case 'getcoords'             % Get current co-ordinates from XYZ widget
    %======================================================================
        % xyz = spm_results_ui('GetCoords',hFxyz)
        if nargin<2, hFxyz='Interactive'; else hFxyz=varargin{2}; end
        hFxyz     = spm_results_ui('FindXYZframe',hFxyz);
        varargout = {getfield(get(hFxyz,'UserData'),'xyz')};
 
 
    %======================================================================
    case 'setcoords'                       % Set co-ordinates to XYZ widget
    %======================================================================
        % [xyz,d] = spm_results_ui('SetCoords',xyz,hFxyz,hC)
        if nargin<4, hC=0; else hC=varargin{4}; end
        if nargin<3, hFxyz=spm_results_ui('FindXYZframe'); else hFxyz=varargin{3}; end
        if nargin<2, error('Set co-ords to what!'); else xyz=varargin{2}; end
 
        %-If this is an internal call, then don't do anything
        if hFxyz==hC, return, end
 
        UD = get(hFxyz,'UserData');
 
        %-Check validity of coords only when called without a caller handle
        %------------------------------------------------------------------
        if hC <= 0
            [xyz,d] = spm_XYZreg('RoundCoords',xyz,UD.M,UD.DIM);
            if d>0 && nargout<2, warning(sprintf(...
                '%s: Co-ords rounded to nearest voxel centre: Discrepancy %.2f',...
                mfilename,d))
            end
        else
            d = [];
        end
 
        %-Update xyz information & widget strings
        %------------------------------------------------------------------
        UD.xyz = xyz; set(hFxyz,'UserData',UD)
        set(UD.hX,'String',sprintf('%.2f',xyz(1)))
        set(UD.hY,'String',sprintf('%.2f',xyz(2)))
        set(UD.hZ,'String',sprintf('%.2f',xyz(3)))
        spm_results_ui('UpdateSPMval',UD)
 
        %-Tell the registry, if we've not been called by the registry...
        %------------------------------------------------------------------
        if (~isempty(UD.hReg) && UD.hReg~=hC)
            spm_XYZreg('SetCoords',xyz,UD.hReg,hFxyz);
        end
 
        %-Return arguments
        %------------------------------------------------------------------
        varargout = {xyz,d};
 

    %======================================================================
    case 'findxyzframe'                                  % Find hFxyz frame
    %======================================================================
        % hFxyz = spm_results_ui('FindXYZframe',h)
        % Sorts out hFxyz handles
        if nargin<2, h='Interactive'; else, h=varargin{2}; end
        if ischar(h), h=spm_figure('FindWin',h); end
        if ~ishandle(h), error('invalid handle'), end
        if ~strcmp(get(h,'Tag'),'hFxyz'), h=findobj(h,'Tag','hFxyz'); end
        if isempty(h), error('XYZ frame not found'), end
        if length(h)>1, error('Multiple XYZ frames found'), end
        varargout = {h};


    %======================================================================
    case 'plotui'                               %-GUI for plot manipulation
    %======================================================================
        % spm_results_ui('PlotUi',hAx)
        if nargin<2, hAx=gca; else hAx=varargin{2}; end
 
        WS = spm('WinScale');
        FS = spm('FontSizes');
        Finter=spm_figure('FindWin','Interactive');
        figure(Finter)
 
        %-Check there aren't already controls!
        %------------------------------------------------------------------
        hGraphUI = findobj(Finter,'Tag','hGraphUI');
        if ~isempty(hGraphUI)           %-Controls exist
            hBs = get(hGraphUI,'UserData');
            if hAx==get(hBs(1),'UserData')  %-Controls linked to these axes
                return
            else                %-Old controls remain
                delete(findobj(Finter,'Tag','hGraphUIbg'))
            end
        end
 
        %-Frames & text
        %------------------------------------------------------------------
        hGraphUIbg = uicontrol(Finter,'Style','Frame','Tag','hGraphUIbg',...
            'BackgroundColor',spm('Colour'),...
            'Position',[001 196 400 055].*WS);
        hGraphUI   = uicontrol(Finter,'Style','Frame','Tag','hGraphUI',...
            'Position',[008 202 387 043].*WS);
        hGraphUIButtsF = uicontrol(Finter,'Style','Frame',...
            'Position',[010 205 380 030].*WS);
        hText = uicontrol(Finter,'Style','Text','String','plot controls',...
            'Position',[020 227 080 016].*WS,...
            'FontWeight','Normal',...
            'FontAngle','Italic','FontSize',FS(10),...
            'HorizontalAlignment','Left',...
            'ForegroundColor','w');
 
        %-Controls
        %------------------------------------------------------------------
        h1 = uicontrol(Finter,'Style','CheckBox','String','hold',...
            'ToolTipString','toggle hold to overlay plots',...
            'FontSize',FS(10),...
            'Value',strcmp(get(hAx,'NextPlot'),'add'),...
            'Callback',[...
            'if get(gcbo,''Value''), ',...
            'set(get(gcbo,''UserData''),''NextPlot'',''add''), ',...
            'else, ',...
            'set(get(gcbo,''UserData''),''NextPlot'',''replace''), ',...
            'end'],...
            'Interruptible','on','Enable','on',...
            'Tag','holdButton',...
            'Position',[015 210 070 020].*WS);
        set(findobj('Tag','plotButton'),'UserData',h1);
 
        h2 = uicontrol(Finter,'Style','CheckBox','String','grid',...
            'ToolTipString','toggle axes grid',...
            'FontSize',FS(10),...
            'Value',strcmp(get(hAx,'XGrid'),'on'),...
            'Callback',[...
            'if get(gcbo,''Value''), ',...
            'set(get(gcbo,''UserData''),''XGrid'',''on'','...
            '''YGrid'',''on'',''ZGrid'',''on''), ',...
            'else, ',...
            'set(get(gcbo,''UserData''),''XGrid'',''off'','...
            '''YGrid'',''off'',''ZGrid'',''off''), ',...
            'end'],...
            'Interruptible','on','Enable','on',...
            'Position',[090 210 070 020].*WS);
        h3 = uicontrol(Finter,'Style','CheckBox','String','Box',...
            'ToolTipString','toggle axes box',...
            'FontSize',FS(10),...
            'Value',strcmp(get(hAx,'Box'),'on'),...
            'Callback',[...
            'if get(gcbo,''Value''), ',...
            'set(get(gcbo,''UserData''),''Box'',''on''), ',...
            'else, ',...
            'set(get(gcbo,''UserData''),''Box'',''off''), ',...
            'end'],...
            'Interruptible','on','Enable','on',...
            'Position',[165 210 070 020].*WS);
        h4 = uicontrol(Finter,'Style','PopUp',...
            'ToolTipString','edit axis text annotations',...
            'FontSize',FS(10),...
            'String','text|Title|Xlabel|Ylabel',...
            'Callback','spm_results_ui(''PlotUiCB'')',...
            'Interruptible','on','Enable','on',...
            'Position',[240 210 070 020].*WS);
        h5 = uicontrol(Finter,'Style','PopUp',...
            'ToolTipString','change various axes attributes',...
            'FontSize',FS(10),...
            'String','attrib|LineWidth|XLim|YLim|handle',...
            'Callback','spm_results_ui(''PlotUiCB'')',...
            'Interruptible','off','Enable','on',...
            'Position',[315 210 070 020].*WS);
 
        %-Handle storage for linking, and DeleteFcns for linked deletion
        %------------------------------------------------------------------
        set(hGraphUI,'UserData',[h1,h2,h3,h4,h5])
        set([h1,h2,h3,h4,h5],'UserData',hAx)
 
        set(hGraphUIbg,'UserData',...
            [hGraphUI,hGraphUIButtsF,hText,h1,h2,h3,h4,h5],...
            'DeleteFcn','spm_results_ui(''Delete'',get(gcbo,''UserData''))')
        set(hAx,'UserData',hGraphUIbg,...
            'DeleteFcn','spm_results_ui(''Delete'',get(gcbo,''UserData''))')


    %======================================================================
    case 'plotuicb'
    %======================================================================
        % spm_results_ui('PlotUiCB')
        hPM = gcbo;
        v   = get(hPM,'Value');
        if v==1, return, end
        str = cellstr(get(hPM,'String'));
        str = str{v};
 
        hAx = get(hPM,'UserData');
        switch str
            case 'Title'
                h = get(hAx,'Title');
                set(h,'String',spm_input('Enter title:',-1,'s+',get(h,'String')))
            case 'Xlabel'
                h = get(hAx,'Xlabel');
                set(h,'String',spm_input('Enter X axis label:',-1,'s+',get(h,'String')))
            case 'Ylabel'
                h = get(hAx,'Ylabel');
                set(h,'String',spm_input('Enter Y axis label:',-1,'s+',get(h,'String')))
            case 'LineWidth'
                lw = spm_input('Enter LineWidth',-1,'e',get(hAx,'LineWidth'),1);
                set(hAx,'LineWidth',lw)
            case 'XLim'
                XLim = spm_input('Enter XLim',-1,'e',get(hAx,'XLim'),[1,2]);
                set(hAx,'XLim',XLim)
            case 'YLim'
                YLim = spm_input('Enter YLim',-1,'e',get(hAx,'YLim'),[1,2]);
                set(hAx,'YLim',YLim)
            case 'handle'
                varargout={hAx};
            otherwise
                warning(['Unknown action: ',str])
        end
 
        set(hPM,'Value',1)
 
 
    %======================================================================
    case 'clear'                                    %-Clear results subpane
    %======================================================================
        % Fgraph = spm_results_ui('Clear',F,mode)
        % mode 1 [default] usual, mode 0 - clear & hide Res stuff, 2 - RNP
 
        if nargin<3, mode=1; else, mode=varargin{3}; end
        if nargin<2, F='Graphics'; else, F=varargin{2}; end
        F = spm_figure('FindWin',F);
 
        %-Clear input objects from 'Interactive' window
        %------------------------------------------------------------------
        %spm_input('!DeleteInputObj')
 
 
        %-Get handles of objects in Graphics window & note permanent results objects
        %------------------------------------------------------------------
        H = get(F,'Children');                          %-Get contents of window
        H = findobj(H,'flat','HandleVisibility','on');  %-Drop GUI components
        h = findobj(H,'flat','Tag','PermRes');          %-Look for 'PermRes' object
 
        if ~isempty(h)
            %-Found 'PermRes' object
            % This has handles of permanent results objects in it's UserData
            tmp  = get(h,'UserData');
            HR   = tmp.H;
            HRv  = tmp.Hv;
        else
            %-No trace of permanent results objects
            HR   = [];
            HRv  = {};
        end
        H = setdiff(H,HR);              %-Drop permanent results obj
 
 
        %-Delete stuff as appropriate
        %------------------------------------------------------------------
        if mode==2  %-Don't delete axes with NextPlot 'add'
            H = setdiff(H,findobj(H,'flat','Type','axes','NextPlot','add'));
        end
 
        delete(H)
 
        if mode==0  %-Hide the permanent results section stuff
            set(HR,'Visible','off')
        else
            set(HR,{'Visible'},HRv)
        end
 
        
    %======================================================================
    case 'close'                                            %-Close Results
    %======================================================================
        spm_clf('Interactive');
        spm_clf('Graphics');
        close(spm_figure('FindWin','Satellite'));
        evalin('base','clear');
    
    
    %======================================================================
    case 'launchmp'                            %-Launch multiplanar toolbox
    %======================================================================
        % hMP = spm_results_ui('LaunchMP',M,DIM,hReg,hBmp)
        if nargin<5, hBmp = gcbo; else hBmp = varargin{5}; end
        hReg = varargin{4};
        DIM  = varargin{3};
        M    = varargin{2};
 
        %-Check for existing MultiPlanar toolbox
        hMP  = get(hBmp,'UserData');
        if ishandle(hMP)
            figure(ancestor(hMP,'figure'));
            varargout = {hMP};
            return
        end
 
        %-Initialise and cross-register MultiPlanar toolbox
        hMP = spm_XYZreg_Ex2('Create',M,DIM);
        spm_XYZreg('Xreg',hReg,hMP,'spm_XYZreg_Ex2');
 
        %-Setup automatic deletion of MultiPlanar on deletion of results controls
        set(hBmp,'Enable','on','UserData',hMP)
        set(hBmp,'DeleteFcn','spm_results_ui(''delete'',get(gcbo,''UserData''))')
 
        varargout = {hMP};
 
 
    %======================================================================
    case 'delete'                           %-Delete HandleGraphics objects
    %======================================================================
        % spm_results_ui('Delete',h)
        h = varargin{2};
        delete(h(ishandle(h)));
 
 
    %======================================================================
    otherwise
    %======================================================================
        error('Unknown action string')
 
end

%==========================================================================
function mychgcon(obj,evt,xSPM)
%==========================================================================
xSPM2.swd   = xSPM.swd;
try, xSPM2.units = xSPM.units; end
xSPM2.Ic    = getfield(get(obj,'UserData'),'Ic');
if isempty(xSPM2.Ic) || all(xSPM2.Ic == 0), xSPM2 = rmfield(xSPM2,'Ic'); end
xSPM2.Im    = xSPM.Im;
xSPM2.pm    = xSPM.pm;
xSPM2.Ex    = xSPM.Ex;
xSPM2.title = '';
if ~isempty(xSPM.thresDesc)
    td = regexp(xSPM.thresDesc,'p\D?(?<u>[\.\d]+) \((?<thresDesc>\S+)\)','names');
    if isempty(td)
        td = regexp(xSPM.thresDesc,'\w=(?<u>[\.\d]+)','names');
        td.thresDesc = 'none';
    end
    if strcmp(td.thresDesc,'unc.'), td.thresDesc = 'none'; end
    xSPM2.thresDesc = td.thresDesc;
    xSPM2.u     = str2double(td.u);
    xSPM2.k     = xSPM.k;
end
hReg = spm_XYZreg('FindReg',spm_figure('GetWin','Interactive'));
xyz  = spm_XYZreg('GetCoords',hReg);
[hReg,xSPM,SPM] = spm_results_ui('setup',xSPM2);
spm_XYZreg('SetCoords',xyz,hReg);
assignin('base','hReg',hReg);
assignin('base','xSPM',xSPM);
assignin('base','SPM',SPM);
figure(spm_figure('GetWin','Interactive'));
```

## spm_ROI_extend.m

```matlab
function [xY, XYZmm, j] = spm_ROI_extend(xY, XYZmm,Input_VOI_path)
% Region of Interest specification
% FORMAT xY = spm_ROI(xY)
% xY     - VOI structure
%    xY.def      - VOI definition [sphere, box, mask, cluster, all]
%    xY.rej      - cell array of disabled VOI definition options
%    xY.xyz      - centre of VOI {mm}
%    xY.spec     - VOI definition parameters
%    xY.str      - description of the VOI
%
% FORMAT [xY, XYZmm, j] = spm_ROI(xY, XYZmm)
% XYZmm  - [3xm] locations of voxels {mm}
%          If an image filename, an spm_vol structure or a NIfTI object is
%          given instead, XYZmm will be initialised to all voxels within
%          the field of view of that image.
%
% XYZmm  - [3xn] filtered locations of voxels {mm} (m>=n) within VOI xY
% j      - [1xn] indices of input locations XYZmm within VOI xY
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston, Guillaume Flandin
% $Id: spm_ROI.m 3960 2010-06-30 17:41:24Z ged $

if nargin < 2 && nargout > 1
    error('Too many output arguments.');
end
%Input_VOI_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\MASK\HG\HG_Resliced_NiftiPairs\NiftiPairs_Resliced_HG.img';
try, xY; catch, xY = []; end

%-Specify ROI
%==========================================================================
if ~isfield(xY,'def')
    def        = {'sphere','box','cluster','mask'};
    if isfield(xY,'rej')
        if ~isfield(xY,'M')
            xY.rej = {xY.rej{:} 'cluster'};
        end
    else
        if isfield(xY,'M')
            xY.rej = {};
        else
            xY.rej = {'cluster'};
        end
    end
    [q, i] = setdiff(def,xY.rej);
    def    = def(sort(i));
    xY.def     = 'mask'; %======'sphere','box','cluster','mask'======================================================
end

%-ROI parameters
%--------------------------------------------------------------------------
switch lower(xY.def)

    case 'sphere'
    %----------------------------------------------------------------------
    if ~isfield(xY,'xyz') || isempty(xY.xyz)
        xY.xyz = spm_input('sphere centre [x y z] {mm}',...
            '!+0','r','0 0 0',3);
    end
    if ~isfield(xY,'spec')
        xY.spec = spm_input('sphere radius (mm)','!+0','r',0,1,[0,Inf]);
    end
    xY.str = sprintf('%0.1fmm sphere',xY.spec);

    case 'box'
    %----------------------------------------------------------------------
    if ~isfield(xY,'xyz') || isempty(xY.xyz)
        xY.xyz = spm_input('box centre [x y z] {mm}',...
            '!+0','r','0 0 0',3);
    end
    if ~isfield(xY,'spec')
        xY.spec = spm_input('box dimensions [x y z] {mm}',...
            '!+0','r','0 0 0',3);
    end
    if length(xY.spec) < 3
        xY.spec = xY.spec(1)*[1 1 1];
    end
    xY.str = sprintf('%0.1f x %0.1f x %0.1f mm box',xY.spec);
    
    case 'mask'
    %----------------------------------------------------------------------
    if ~isfield(xY,'spec')
        xY.spec = spm_vol([Input_VOI_path,',1']);%===========================================================
    else
        if ~isstruct(xY.spec)
            xY.spec = spm_vol(xY.spec);
        end
    end
    str    = spm_str_manip(xY.spec.fname,'a30x');
    xY.str = sprintf('image mask: %s',str); 
        
    case 'cluster'
    %----------------------------------------------------------------------
    if ~isfield(xY,'xyz') || isempty(xY.xyz)
        xY.xyz = spm_input('seed voxel [x y z] {mm}',...
            '!+0','r','0 0 0',3);
    end
    if ~isfield(xY,'M')
        xY.M = spm_input('affine transformation matrix',...
            '!+0','r','0 0 0',[4 4]);
    end
    xY.spec = [];
    xY.str  = sprintf('cluster (seed voxel: %0.1f %0.1f %0.1f)',xY.xyz);
    
    case 'all'
    %----------------------------------------------------------------------
    xY.str  = 'all';
    
    otherwise
    %----------------------------------------------------------------------
    error('Unknown VOI type.');
    
end

if nargin < 2, return; end

%-'Estimate' ROI
%==========================================================================

%-Argument check
%--------------------------------------------------------------------------
if ischar(XYZmm) && isempty(XYZmm)
    XYZmm = spm_select(1,'image','Specify Image');
end
if ischar(XYZmm), XYZmm = spm_vol(XYZmm); end
if isa(XYZmm,'nifti')
    XYZmm    = struct('dim',size(XYZmm.dat), 'mat',XYZmm.mat);
end
if isstruct(XYZmm) % spm_vol
    [R,C,P]  = ndgrid(1:XYZmm.dim(1),1:XYZmm.dim(2),1:XYZmm.dim(3));
    RCP      = [R(:)';C(:)';P(:)'];
    clear R C P
    RCP(4,:) = 1;
    XYZmm    = XYZmm.mat(1:3,:)*RCP;    
end
if isempty(XYZmm), XYZmm = zeros(3,0); end

%-Filter location of voxels
%--------------------------------------------------------------------------
Q          = ones(1,size(XYZmm,2));

switch lower(xY.def)

    case 'sphere'
    %----------------------------------------------------------------------
    j      = find(sum((XYZmm - xY.xyz*Q).^2) <= xY.spec^2);

    case 'box'
    %----------------------------------------------------------------------
    j      = find(all(abs(XYZmm - xY.xyz*Q) <= xY.spec(:)*Q/2));
    
    case 'mask'
    %----------------------------------------------------------------------
    XYZ    = xY.spec.mat \ [XYZmm; Q];
    j      = find(spm_sample_vol(xY.spec, XYZ(1,:), XYZ(2,:), XYZ(3,:),0) > 0);
    
    case 'cluster'
    %----------------------------------------------------------------------
    [x i]  = spm_XYZreg('NearestXYZ',xY.xyz,XYZmm);
    XYZ    = round(xY.M \ [XYZmm; Q]);
    A      = spm_clusters(XYZ);
    j      = find(A == A(i));
    
    case 'all'
    %----------------------------------------------------------------------
    j      = 1:size(XYZmm,2);
    
    otherwise
    %----------------------------------------------------------------------
    error('Unknown VOI type.');
    
end

XYZmm      = XYZmm(:,j);
if strcmpi(xY.def,'mask') && ~isempty(XYZmm), xY.xyz = mean(XYZmm,2); end
```

## createVOI

```matlab
function createVOI(spm_mat_path,Input_u)
% 功能： 使用特定的Mask抽取VOI。
% spm_mat_path : SPM.mat的完整路径，需要是做完FirstLevel之后的SPM.mat;
% Input_u : 抽取时间序列时使用的P值，一般默认是0.001，当抽取失败的时候，，适当调大可以确保成功；
% -----------------------------------------------------------------------------------------
% 配置信息：
% contrast_name ： First_Level时候设置的contrast,这里的顺序很重要，因为程序中使用数字表示每个contrast的；
% InputMask ： appying mask : 0 none ; 1 contrast ; 2 image ;默认是整数 0 
% InputthresDesc : p value adjustment to control: 'FWE' or 'none'
% Input_k : extend threshold {voxel}  0 ; 默认是0
% xx_mask_path : 抽取的VOI使用的mask的绝对路径
% VOI_Mask ：抽取的VOI使用的mask的绝对路径
% Input_is ： adjust_contrst 这里选择那个F-All的,整数，1 是dont adjust, 2 是F-All，3个数字分别对应三个Mask的adjust_contrst。
contrast_name = {'F-All','JX','DW','RL','ZR'};
InputMask = 0;  %appying mask : 0 none ; 1 contrast ; 2 image ============================================================================
InputthresDesc = 'none';  %p value adjustment to control: 'FWE' or 'none'=================================================================
Input_k = 0; % extend threshold {voxel}  0 ===============================================================================================
HG_mask_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\MASK\HG\HG_Resliced_NiftiPairs\NiftiPairs_Resliced_HG.img';
MFG_mask_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\MASK\MFG\MFG_Resliced_NiftiPairs\NiftiPairs_Resliced_MFG.img';
STG_mask_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\MASK\STG\STG_Resliced_NiftiPairs\NiftiPairs_Resliced_STG.img';
VOI_Mask = {HG_mask_path,MFG_mask_path,STG_mask_path};
Input_is = {2,2,2}; %adjust_contrst 这里选择那个F-All的,整数，1 是dont adjust, 2 是F-All.
empty_bit = []; % 占位符，没有用处
% 配置结束
% ----------------------------------------------------------------------------------------------------------------------------------------
for j = 2:size(contrast_name,2)
    Ic = j;  % 要使用的Contrast的编号，这里1 : F-All  2: JX 3: DW 4:RL 5:ZR
    VOI_Names = {[contrast_name{Ic},'_HG'],[contrast_name{Ic},'_MFG'],[contrast_name{Ic},'_STG',]}; %Input_VOIName : 抽取的VOI的名称
    [SPM,xSPM] = spm_getSPM_extend(spm_mat_path,Ic,InputMask,InputthresDesc,Input_u,Input_k);
    [hReg,xSPM,SPM] = spm_result_ui_extend('Setup',xSPM);
    for i = 1:size(VOI_Mask,2)
        [Y xY] = spm_regions_extend(xSPM,SPM,hReg,empty_bit,VOI_Names{i},Input_is{i},VOI_Mask{i});
    end
    clear SPM;
    clear xSPM;
end
```

## createVOIs, 创建多个被试的ROI
```matlab
% 创建多个被试的VOI；
% 
first_level_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\';
cd(first_level_path);
dir_str = dir('2016*');
Input_u = 0.001; % P值；抽取不成功时，调大P值可以成功
for i = 1:size(dir_str,1)
    spm_mat_path = [first_level_path,dir_str(i).name,'\SPM.mat'];
    createVOI(spm_mat_path);
end

```

## create_dcm
```matlab
function DCM = create_dcm(subject_path)
%功能： 定义DCM模型，需要先做完抽取VOI，在FirstLevel文件夹下面VOI_开头的文件；
%subject_path : First_Level 被试目录， eg.D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\20160716002\
%condition_name : SPM.mat 设计矩阵中定义的条件，这里用来给生成的DCM命名。
%---------------------------------------------------------------
%-配置----------------------------------------------------------
condition_name = {'JX','DW','RL','ZR'};
%Input_a : DCM模型矩阵，需要更改模型的时候，修改这个矩阵
%Input_b : 调节输入
%Input_c : 外界输入
%-配置结束------------------------------------------------------
cd(subject_path);
spmmatfile = [subject_path,'SPM.mat'];
for i = 1:size(condition_name,2)  %每次循环创建一个condition条件下的DCM模型
    name = condition_name{i}; % 生成的DCM模型的名称；
    condition_mask = [0,0,0,0];
    condition_mask(i) = 1; % 使用哪个condition作为
    TE = 0.04; %  TE 
    Input_a = [1,1,1;1,1,1;1,1,1]; % 定义DCM模型的连接矩阵
    Input_b = [0,0,0;0,0,0;0,0,0]; % 定义调节参数
    Input_c = [1;0;0]; % 定义输入参数
    
    % 获得VOI
    %------------------------------------------------
    filter = ['VOI_',condition_name{i},'_*'];
    VOIs_path = dir(filter);
    VOIs = cell(size(VOIs_path,1),1);
    for j = 1: size(VOIs_path,1)
        VOIs{j} = [subject_path,VOIs_path(j).name];
    end
    %-------------------------------------------------
    DCM = spm_dcm_specify_extend(spmmatfile,name,VOIs,condition_mask,TE,Input_a,Input_b,Input_c);
    clear name;
    clear VOIs;
    clear condition_mask;
end
```

## create_dcms, 定义多个被试的DCM模型
```matlab
%配置：
first_level_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\';
cd(first_level_path);
dir_str = dir('2016*');

for i = 1:size(dir_str,1)
    subject_path = [first_level_path,dir_str(i).name,'\'];
    create_dcm(subject_path);
end
```

## 批量估计DCM模型

```matlab
% 估计DCM模型
first_level_path = 'D:\FMRI_ROOT\YANTAI\ANALYSIS\first_level_dcm_w_whole\';
cd(first_level_path);
dir_path = dir('2016*');
for i = 1: size(dir_path,1)
    subject_path = [first_level_path,dir_path(i).name];
    cd(subject_path);
    dcm_models_path = dir('DCM*');
    for j = 1:size(dcm_models_path,1)
       dcm_model_path =  [subject_path,'\',dcm_models_path(j).name];
       spm_dcm_estimate(dcm_model_path);
    end
end
```
