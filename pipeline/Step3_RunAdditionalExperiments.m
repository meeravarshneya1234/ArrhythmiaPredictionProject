%% Load Saved Population Data
load('TestPop\BasePop.mat')

%% Run High and Low Cao Experiments 
% cardiomyocyte stimulation settings 
settings_Cao.celltype = 'endo';
settings_Cao.PCL = 1000 ;  % Pacing, Interval bewteen stimuli,[ms]
settings_Cao.stim_delay = 100 ; % Time the first stimulus, [ms]
settings_Cao.stim_dur = 2 ; % Stimulus duration
settings_Cao.stim_amp = 32.2; % Stimulus amplitude 

% variability settings 
settings_Cao.scalings = popscalings; % use saved original population scalings 
settings_Cao.ICs = popICs; % use saved original population ICs 
settings_Cao.variations = size(popscalings,1); % population size 

% calibrate population? 
% when creating the initial population, sometimes certain parameter sets
% create cells that form arrhythmic activity before any trigger is applied.
% we removed those cells, and reran new ones.
settings_Cao.remove_arrhythmias = false; % do not remove cells that form arrhythmic activity 
settings_Cao.remove_experimental = false; % do not remove cells not within experimental range
settings_Cao.reruncells = false; % do not rerun removed cells 

%-- Run Low Cao 
settings_Cao.Folder = 'TestPop/LowCao';
mkdir(settings_Cao.Folder)
pert = settings_blockcurrents; 
pert.Cao = 0.5;
BuildPopulation(settings_Cao,pert)

%-- Run High Cao
settings_Cao.Folder = 'TestPop/HighCao';
mkdir(settings_Cao.Folder)
pert = settings_blockcurrents; 
pert.Cao = 2;
BuildPopulation(settings_Cao,pert)

%% Run Threshold Experiment
settings_thres.celltype = 'endo'; % 'epi', 'endo', 'mid',
settings_thres.PCL =1000 ;  % Interval bewteen stimuli,[ms]
settings_thres.stim_delay = 100 ; % Time the first stimulus, [ms]
settings_thres.stim_dur = 2 ; % Stimulus duration
settings_thres.stim_amp = 32.2; % Stimulus amplitude
settings_thres.numbertokeep = 1 ;% Determine how many beats to keep. 1 = last beat, 2 = last two beats
settings_thres.steady_state = 0;
settings_thres.variations = 1; 

% Make sure this is set to 1 because we're already at steady state 
settings_thres.nBeats = 1 ; % Number of beats to simulate

n_variants = size(popscalings,1);
plot_flag = false; % plot the results
threshold = zeros(n_variants,1);
for i = 1:n_variants
    settings_thres.scalings = popscalings(i,:); % use saved original population scalings
    settings_thres.ICs = popICs(i,:); % use saved original population ICs
    threshold(i) = Find_Threshold(settings_thres,plot_flag);
end

savdir = 'TestPop/';
path = fullfile(savdir,'Thresholds.mat');
save(path,'threshold')
%% Run Fast and Slow Rates Experiments 
% cardiomyocyte stimulation settings 
settings_Rate.celltype = 'endo';
settings_Rate.stim_delay = 100 ; % Time the first stimulus, [ms]
settings_Rate.stim_dur = 2 ; % Stimulus duration
settings_Rate.stim_amp = 32.2; % Stimulus amplitude 

% variability settings 
settings_Rate.scalings = popscalings; % use saved original population scalings 
settings_Rate.ICs = popICs; % use saved original population ICs 
settings_Rate.variations = size(popscalings,1); % population size 

% calibrate population? 
% when creating the initial population, sometimes certain parameter sets
% create cells that form arrhythmic activity before any trigger is applied.
% we removed those cells, and reran new ones.
settings_Rate.remove_arrhythmias = false; % do not remove cells that form arrhythmic activity 
settings_Rate.remove_experimental = false; % do not remove cells not within experimental range
settings_Rate.reruncells = false; % do not rerun removed cells 

%-- Run Fast Pacing
settings_Rate.Folder = 'TestPop/PCL_400';
mkdir(settings_Rate.Folder)
settings_Rate.PCL = 400 ;  % Pacing, Interval bewteen stimuli,[ms]
pert = settings_blockcurrents; 
BuildPopulation(settings_Rate,pert)

%-- Run Slow Pacing
settings_Rate.Folder = 'TestPop/PCL_5000';
mkdir(settings_Rate.Folder)
settings_Rate.PCL = 5000 ;  % Pacing, Interval bewteen stimuli,[ms]
pert = settings_blockcurrents; 
BuildPopulation(settings_Rate,pert)