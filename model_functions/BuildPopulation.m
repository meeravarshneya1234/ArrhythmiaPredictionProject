function BuildPopulation(settings,pert)
%--------------------------------------------------------------------------
%% -- BuildPopulation.m -- %%
% Description: Build population

% Inputs:
% --> settings - [struct] simulation protocol settings
% --> pert  - [struct] initialize channel block settings

% Outputs: No actual outputs. Data is automatically saved to folder.
% -------------------------------------------------------------------------
%% Settings
settings.nBeats = 10; % Number of beats to simulate
settings.numbertokeep = 10 ;% Determine how many beats to keep. 1 = last beat, 2 = last two beats
settings.steady_state = true; % Start with each scenario's BL model steady state values.

%% Run Simulation

disp('Running New Population...')

% Does the user provide a parameter matrix or should we make one?
if ~isfield(settings,'scalings') % make new one
    settings.scalings = popfuncs.create_scale_vector(settings,settings.variations);
end

% Separate population
intervals = chunks(settings.variations);
chunk_settings = settings;

% Create variables to save just the final ICs ans parameter matrix.
popICs = [];
popscalings = [];

% Loop through multiple intervals of the data
for ii = 1:size(intervals,1)
    chunk_settings.scalings = settings.scalings(intervals(ii,1):intervals(ii,2),:);
    [chunk_settings.variations,~] = size(chunk_settings.scalings);

    X = runSim(chunk_settings,pert); % run simulation
    
    % clean data? 
    if settings.remove_arrhythmias || settings.remove_experimental
        X = popfuncs.clean_population(X,chunk_settings);
    end
    
    % save the ICs and scalings
    for i = 1:length(X)
        popICs(end+1,:) = X(i).states(end,:);
        popscalings(end+1,:) = X(i).scalings;
    end
    
    if settings.variations > 1000
        % Save the data to Folder and then delete matrix to save space
        matfile = fullfile(PopFolder, ['Population_' num2str(ii) '.mat']);
        save(matfile, 'X')
        clear X
    end
    disp([num2str((ii/(size(intervals,1)))*100) '% Finished '])

end
%% Run Population using final steady state ICs 
settings2.celltype = settings.celltype;
settings2.PCL = settings.PCL ;  % Pacing, Interval bewteen stimuli,[ms]
settings2.stim_delay = settings.stim_delay ; % Time the first stimulus, [ms]
settings2.stim_dur = settings.stim_dur ; % Stimulus duration
settings2.stim_amp = settings.stim_amp ; % Stimulus amplitude 
settings2.nBeats = 1 ; % Only one beat
settings2.numbertokeep = 1 ;% Save only that one beat
settings2.ICs = popICs;
settings2.scalings = popscalings;
settings2.variations = size(popscalings,1);

% Run each cell in population with new steady state ICs 
BaseCells = runSim(settings2,pert); % run simulation

figure
for ii = 1:settings2.variations
    subplot(1,2,1)
    plot(BaseCells(ii).times,BaseCells(ii).states(:,1),'linewidth',2)
    hold on
    set(gca,'FontSize',14,'FontWeight','bold','FontName','Calibri','XGrid','on','YGrid','on')
    xlabel('time (ms)','FontSize',14,'FontWeight','bold','FontName','Calibri')
    ylabel('Voltage (mV)','FontSize',14,'FontWeight','bold','FontName','Calibri')

    subplot(1,2,2)
    plot(BaseCells(ii).times,BaseCells(ii).states(:,6),'linewidth',2)
    hold on
    set(gca,'FontSize',14,'FontWeight','bold','FontName','Calibri','XGrid','on','YGrid','on')
    xlabel('time (ms)','FontSize',14,'FontWeight','bold','FontName','Calibri')
    ylabel('Cai (uM)','FontSize',14,'FontWeight','bold','FontName','Calibri')
end
set(gcf,'Position',[680 427 299 551])
picfile = fullfile(settings.Folder, 'Population.png');
saveas(gcf,picfile)

matfile = fullfile(settings.Folder, 'BasePop.mat');
save(matfile, 'BaseCells','popICs','popscalings')

%% -- Nested Functions 
function intervals = chunks(variations)
    if variations > 1000
        ints = round(linspace(0,variations,variations/500));
        intervals = zeros((length(ints) - 1),2);
        for i = 1:(length(ints) - 1)
            intervals(i,1:2) = [ints(i)+1 ints(i+1)];
        end
        
        % Create a Folder to store all of the population data.
        PopFolder = [settings.Folder '\Population\'];
        mkdir(PopFolder)
    else
        intervals = [1 variations];
    end