%% SFigure 3A
load('Manuscript_Data\BasePop') %load population data

settings.celltype = 'endo';
settings.PCL = 1000 ;  % Pacing, Interval bewteen stimuli,[ms]
settings.stim_delay = 100 ; % Time the first stimulus, [ms]
settings.stim_dur = 2 ; % Stimulus duration
settings.stim_amp = 32.2; % Stimulus amplitude
settings.numbertokeep = 1;
settings.nBeats = 1;
settings.variations = 1;
settings.sigmaG = 0;
settings.sigmap = 0;
settings.sigmaV = 0;
settings.steady_state = 1;

% Run cells without trigger
pert = settings_blockcurrents;
datatable = runSim(settings,pert);

t = datatable.times;
V = datatable.states(:,1);
Cai = datatable.states(:,6);

figure
plot(t,V,'linewidth',2)
hold on

% Run cells with slow (5000 ms) and fast (400 ms) pacing 
settings.nBeats = 100;
PCLs = [5000 400];
for i = 1:2
    
    settings.PCL = PCLs(i);
    pert = settings_blockcurrents;
    datatable = runSim(settings,pert);
    
    t = datatable.times;
    V = datatable.states(:,1); 
    plot(t,V,'linewidth',2)
    
end
xlabel('time (ms)','FontSize',12,'FontWeight','bold','FontName','Calibri')
ylabel('Voltage (mV)','FontSize',12,'FontWeight','bold','FontName','Calibri')
legend ('Steady State','Slow (0.2 Hz)','Fast (2.5 Hz)')
xlim([0 1000])
set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')

%% SFigure 3B-C

labels = {'SteadyState','SteadyState+SlowAPFeats','SteadyState+FastAPFeats','SteadyState+PacingAPFeats'};
%% IKr Block Prediction Performance 
datasets = {'SteadyState_APFeatures_IKrBlock'...
    'SteadyState_APFeatures_Slow_APFeatures_IKrBlock'...
    'SteadyState_APFeatures_Fast_APFeatures_IKrBlock'...
    'SteadyState_APFeatures_Pacing_APFeatures_IKrBlock'};
plotML(datasets,labels)
%% ICaL Increase Prediction Performance 
datasets = {'SteadyState_APFeatures_ICaLIncrease'...
    'SteadyState_APFeatures_Slow_APFeatures_ICaLIncrease'...
    'SteadyState_APFeatures_Fast_APFeatures_ICaLIncrease'...
    'SteadyState_APFeatures_Pacing_APFeatures_ICaLIncrease'};
plotML(datasets,labels)
