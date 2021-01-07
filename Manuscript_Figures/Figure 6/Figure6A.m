settings.celltype = 'endo'; % 'epi', 'endo', 'mid',
settings.PCL =1000 ;  % Interval bewteen stimuli,[ms]
settings.stim_delay = 100 ; % Time the first stimulus, [ms]
settings.stim_dur = 2 ; % Stimulus duration
settings.stim_amp = 32.2; % Stimulus amplitude
settings.numbertokeep = 1 ;% Determine how many beats to keep. 1 = last beat, 2 = last two beats
settings.steady_state = 1;

% running only one cell - the baseline model, no variation 
settings.sigmaV = 0;
settings.sigmaG = 0;
settings.sigmap = 0;
settings.variations = 1;

% Make sure this is set to 1 because we're already at steady state 
settings.nBeats = 1 ; % Number of beats to simulate

plot_flag = true; % plot the results
threshold = Find_Threshold(settings,plot_flag);
