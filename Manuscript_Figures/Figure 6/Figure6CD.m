%% Get Threshold Data 
file = 'Manuscript_Data\Population_Feature_Outputs.csv';
data = readtable(file);
feature.data = data.Threshold;

labels = {'Threshold','Steady State','Steady State+Threshold'};
%% Figure 6C - IKr Block Prediction Performance 
datasets = {'SteadyState_APFeatures_IKrBlock'...
    'SteadyState_APFeatures_Threshold_IKrBlock'};
feature.label = data.IKrBlock_Label;
plotML(datasets,labels,feature)
%% Figure 6D - ICaL Increase Prediction Performance 
datasets = {'SteadyState_APFeatures_ICaLIncrease'...
    'SteadyState_APFeatures_Threshold_ICaLIncrease'};
feature.label = data.ICaLIncrease_Label;
plotML(datasets,labels,feature)
