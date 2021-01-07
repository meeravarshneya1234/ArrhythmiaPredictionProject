%% Universal Settings for Building Populations 

% cardiomyocyte stimulation settings 
settings.celltype = 'endo';
settings.PCL = 1000 ;  % Pacing, Interval bewteen stimuli,[ms]
settings.stim_delay = 100 ; % Time the first stimulus, [ms]
settings.stim_dur = 2 ; % Stimulus duration
settings.stim_amp = 32.2; % Stimulus amplitude 

% variability settings 
settings.variations = 1000; % population size 
settings.sigmaG = 0.3; % standard deviation to vary conductances 
settings.sigmap = 0.3; % standard deviation to vary gating variables 
settings.sigmaV = 4; % standard deviation to vary voltage dependences

% calibrate population? 
% when creating the initial population, sometimes certain parameter sets
% create cells that form arrhythmic activity before any trigger is applied.
% we removed those cells, and reran new ones.
settings.remove_arrhythmias = true; % remove cells that form arrhythmic activity 
settings.remove_experimental = true; % remove cells not within experimental range
settings.reruncells = true; % rerun removed cells 

%-- Folder to Save Data 
settings.Folder = 'TestPop';
if ~exist(settings.Folder, 'dir')
    mkdir(yourFolder)
end

%-- Run Population 
pert = settings_blockcurrents; 
BuildPopulation(settings,pert)
