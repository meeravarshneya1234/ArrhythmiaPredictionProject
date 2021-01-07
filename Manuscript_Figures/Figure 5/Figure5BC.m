labels = {'Steady State','Low Cao','High Cao','Combined'};
%% Figure 5B - IKr Block Prediction Performance 
datasets = {'SteadyState_APFeatures_IKrBlock'...
    'SteadyState_APFeatures_LowCao_APFeatures_IKrBlock'...
    'SteadyState_APFeatures_HighCao_APFeatures_IKrBlock'...
    'SteadyState_APFeatures_Cao_APFeatures_IKrBlock'};
plotML(datasets,labels)
%% Figure 5C - ICaL Increase Prediction Performance 
datasets = {'SteadyState_APFeatures_ICaLIncrease'...
    'SteadyState_APFeatures_LowCao_APFeatures_ICaLIncrease'...
    'SteadyState_APFeatures_HighCao_APFeatures_ICaLIncrease'...
    'SteadyState_APFeatures_Cao_APFeatures_ICaLIncrease'};
plotML(datasets,labels)
