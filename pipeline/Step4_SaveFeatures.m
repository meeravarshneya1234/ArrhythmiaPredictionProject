load('TestPop\BasePop.mat') % original pre-trigger population data
%% Steady State Features
outputs_base = zeros(length(BaseCells),14);
for i = 1:length(BaseCells)
    t = BaseCells(i).times;
    V = BaseCells(i).states(:,1);
    Cai = BaseCells(i).states(:,6);
    
    [outputs_base(i,:),outputlabels] = calculate_features(V,Cai,t);    
end
settings.numbertokeep = 1;
settings.stim_delay = 100;
%% High Cao Features
load('TestPop\HighCao\BasePop.mat') % high Cao experiment data
settings.PCL = 1000;
outputs_highcao = zeros(length(BaseCells),14);    
remove_AP = popfuncs.find_arrhythmias(settings,BaseCells);    
for i = 1:length(BaseCells)
    if remove_AP(i)
        outputs_highcao(i,:) = NaN;
    else
        t = BaseCells(i).times;
        V = BaseCells(i).states(:,1);
        Cai = BaseCells(i).states(:,6);   
        [outputs_highcao(i,:),~] = calculate_features(V,Cai,t);
    end
end
outputs_labels_highcao = cellfun(@(c)[c '_HighCao'],outputlabels,'uni',false);

%% Low Cao Features
load('TestPop\LowCao\BasePop.mat') % low Cao experiment data
settings.numbertokeep = 1;
settings.PCL = 1000;
outputs_lowcao = zeros(length(BaseCells),14);
remove_AP = popfuncs.find_arrhythmias(settings,BaseCells);    
for i = 1:length(BaseCells)
    if remove_AP(i)
        outputs_lowcao(i,:) = NaN;
    else
        t = BaseCells(i).times;
        V = BaseCells(i).states(:,1);
        Cai = BaseCells(i).states(:,6);
        [outputs_lowcao(i,:),outputlabels] = calculate_features(V,Cai,t);
    end
end
outputs_labels_lowcao = cellfun(@(c)[c '_LowCao'],outputlabels,'uni',false);

%% Load Threshold Data 
load('TestPop\Thresholds.mat') 
outputs_thresholds = threshold;
outputs_labels_thres = {'Threshold'};

%% Fast Features
load('TestPop\Fast\BasePop.mat') % high Cao experiment data
settings.numbertokeep = 1;
settings.PCL = 400;
outputs_fast = zeros(length(BaseCells),14);    
remove_AP = popfuncs.find_arrhythmias(settings,BaseCells);    
for i = 1:length(BaseCells)
    if remove_AP(i)
        outputs_fast(i,:) = NaN;
    else
        t = BaseCells(i).times;
        V = BaseCells(i).states(:,1);
        Cai = BaseCells(i).states(:,6);   
        [outputs_fast(i,:),~] = calculate_features(V,Cai,t);
    end
end
outputs_labels_fast = cellfun(@(c)[c '_Fast'],outputlabels,'uni',false);

%% Slow Features
load('TestPop\Slow\BasePop.mat') % low Cao experiment data
settings.numbertokeep = 1;
settings.PCL = 5000;
outputs_slow = zeros(length(BaseCells),14);
remove_AP = popfuncs.find_arrhythmias(settings,BaseCells);    
for i = 1:length(BaseCells)
    if remove_AP(i)
        outputs_slow(i,:) = NaN;
    else
        t = BaseCells(i).times;
        V = BaseCells(i).states(:,1);
        Cai = BaseCells(i).states(:,6);
        [outputs_slow(i,:),outputlabels] = calculate_features(V,Cai,t);
    end
end
outputs_labels_slow = cellfun(@(c)[c '_Slow'],outputlabels,'uni',false);
%% Load Trigger Labels 
load('TestPop\IKrBlock\Y.mat') 
Y_IKr = Y;
load('TestPop\ICaLIncrease\Y.mat') 
Y_ICaL = Y;
load('TestPop\Inject\Y.mat') 
Y_Inject = Y;
outputs_labels_labels = {'IKrBlock_Label','ICaLIncrease_Label','Inject_Label'};

%% Concatenate all data 
output = [outputs_base outputs_highcao outputs_lowcao outputs_thresholds outputs_fast outputs_slow Y_IKr Y_ICaL Y_Inject];
outputlabels = [outputlabels outputs_labels_highcao outputs_labels_lowcao outputs_labels_thres outputs_labels_fast outputs_labels_slow outputs_labels_labels];

Output_File = 'TestPop_Features.csv';
Output_Dir = 'C:\TestPop\';
path = fullfile(Output_Dir,Output_File);

if isfile(path)
    disp('FILE ALREADY EXISTS.')
else
    fid = fopen(Output_File, 'w') ;
    fprintf(fid, '%s,', outputlabels{1,1:end-1}) ;
    fprintf(fid, '%s\n', outputlabels{1,end}) ;
    fclose(fid) ;
    dlmwrite(Output_File, output, '-append') ;
end