%% Figure 6B: Find 10 cells with similar APDs but different thresholds 
file = 'Data\Population_Feature_Outputs.csv';
data = readtable(file);

label = data.IKrBlock_Label; thres = data.Threshold;
resistant_dex = find(label==0); susceptible_dex = find(label==1);
[I,J] = sort(thres);
L = label(J); 
APD = data.APD90(J);

get_APD1 = find(APD > 200 & APD < 300 & L == 1 & thres < 30);
get_APD2 = find(APD > 200 & APD < 300 & L == 0 & thres > 36);

pos = randi(length(get_APD1),10,1);
top10_susceptible = get_APD1(pos);
top10_resistant = get_APD2(pos);

%-------------------------------------------------------------------------%
%% Run simulation for these cells 
load('Data\BaseCells.mat');

%%---------------- Run simulation for susceptible cells -----------------%%
settings.celltype = 'endo'; % 'epi', 'endo', 'mid',
settings.PCL =1000 ;  % Interval bewteen stimuli,[ms]
settings.stim_delay = 100 ; % Time the first stimulus, [ms]
settings.stim_dur = 2 ; % Stimulus duration
settings.stim_amp = 28.5; % Stimulus amplitude
settings.nBeats = 1 ; % Number of beats to simulate
settings.numbertokeep = 1 ;% Determine how many beats to keep
settings.steady_state = 0;

% Set Up Population Variants
pert = settings_blockcurrents;
settings.ICs = popICs(top10_susceptible,:);
settings.scalings = popscalings(top10_susceptible,:);
settings.variations = 10;

% Baseline cells
X1 = runSim(settings,pert); % run simulation
figure
for i = 1:settings.variations
    t = X1(i).times;
    V = X1(i).states(:,1);
    
    figure(gcf)
    hold on
    plot(t,V,'linewidth',2)
end
ylim([-100 50])

% Apply Threshold on suscetible cells 
pert = settings_blockcurrents;
pert.GNa = 0;
X2 = runSim(settings,pert); % run simulation
figure
for i = 1:10
    t = X2(i).times;
    V = X2(i).states(:,1);
    
    figure(gcf)
    hold on
    plot(t,V,'linewidth',2)
end
ylim([-100 50])
          
%-------------------------------------------------------------------------%
%%----------------- Run simulation for resistant cells ------------------%%
pert = settings_blockcurrents;
settings.ICs = popICs(top10_resistant,:);
settings.scalings = popscalings(top10_resistant,:);
settings.variations = 10;

% Baseline cells
X3 = runSim(settings,pert); % run simulation
figure
for i = 1:10
    t = X3(i).times;
    V = X3(i).states(:,1);
    
    figure(gcf)
    hold on
    plot(t,V,'linewidth',2)
end
ylim([-100 50])

% Apply Threshold on resistant cells 
pert = settings_blockcurrents;
pert.GNa = 0;

X4 = runSim(settings,pert); % run simulation
figure
for i = 1:10
    t = X4(i).times;
    V = X4(i).states(:,1);
    
    figure(gcf)
    hold on
    plot(t,V,'linewidth',2)
end
ylim([-100 50])