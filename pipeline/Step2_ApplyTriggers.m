%% Universal Settings for Applying Triggers 

% cardiomyocyte settings
settings.celltype = 'endo'; % 'epi', 'endo', 'mid',
settings.PCL = 1000 ;  % Interval bewteen stimuli,[ms]
settings.stim_delay = 100 ; % Time the first stimulus, [ms]
settings.stim_dur = 2 ; % Stimulus duration
settings.stim_amp = 32.2; % Stimulus amplitude 
settings.Folder = 'TestPop/';

%-- Apply IKr Block Trigger
settings.SubFolder = 'IKrBlock';
pert = settings_blockcurrents; 
pert.GKr = 0.06;
ApplyTrigger(settings,pert)

%-- Apply ICaL Increase Trigger
settings.SubFolder = 'ICaLIncrease';
pert = settings_blockcurrents; 
pert.PCa = 15.13;
ApplyTrigger(settings,pert)

%-- Apply Inject Trigger
settings.SubFolder = 'Inject';
pert = settings_blockcurrents; 
pert.Inject = 0.7;
ApplyTrigger(settings,pert)