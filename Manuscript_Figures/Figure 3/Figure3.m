%% Get APD Data 
file = 'Data\Population_Feature_Outputs.csv';
data = readtable(file);
feature.data = data.APD90;

%%%%% ---- %%%%%%
labels = {'APD90','8 AP Feats','6 CaT Feats','14 Feats'};
%% Figure 3B - IKr Block Prediction Performance 
datasets = {'SteadyState_APFeatures_IKrBlock'...
    'SteadyState_CaTFeatures_IKrBlock'...
    'SteadyState_14Features_IKrBlock'};
feature.label = data.IKrBlock_Label;
plotML(datasets,labels,feature)
%% Figure 3C - ICaL Increase Prediction Performance 
datasets = {'SteadyState_APFeatures_ICaLIncrease'...
    'SteadyState_CaTFeatures_ICaLIncrease'...
    'SteadyState_14Features_ICaLIncrease'};
feature.label = data.ICaLIncrease_Label;
plotML(datasets,labels,feature)
%% Figure 3D - Current Inject Prediction Performance 
datasets = {'SteadyState_APFeatures_Inject'...
    'SteadyState_CaTFeatures_Inject'...
    'SteadyState_14Features_Inject'};
feature.label = data.Inject_Label;
plotML(datasets,labels,feature)
