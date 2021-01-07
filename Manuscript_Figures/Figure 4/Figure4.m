labels = {'Conductances+Kinetics','Conductances'};
%% Figure 4A - IKr Block Prediction Performance 
datasets = {'SteadyState_APFeatures_IKrBlock'...
    'Gspop_SteadyState_APFeatures_IKrBlock'};
plotML(datasets,labels)
%% Figure 4B - ICaL Increase Prediction Performance 
datasets = {'SteadyState_APFeatures_ICaLIncrease'...
    'Gspop_SteadyState_APFeatures_ICaLIncrease'};
plotML(datasets,labels)
