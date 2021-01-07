function ApplyTrigger(settings,pert)

%% Standard Settings
settings.nBeats = 10 ; % Number of beats to simulate
settings.numbertokeep = 10 ;% Determine how many beats to keep. 1 = last beat, 2 = last two beats

%% Load ICs from Folder 
File = fullfile(settings.Folder,'BasePop.mat');
load(File,'popscalings','popICs');
[variations,~] = size(popscalings);

%% Set Intervals 
% Since populations are so large, we need to divide how we save the data
% into intervals. Number of sections is determined by input.
intervals = chunks(variations);
chunk_settings = settings;

yourFolder = fullfile(settings.Folder,settings.SubFolder); 
if ~exist(yourFolder, 'dir')
    mkdir(yourFolder)
end

%% Run Simulation
Y = [];
for ii = 1:size(intervals,1)
    % Set Up Population Variants
    n = intervals(ii,1):intervals(ii,2);
    chunk_settings.ICs = popICs(n,:);
    chunk_settings.scalings = popscalings(n,:);
    chunk_settings.variations = length(n);
    
    X = runSim(chunk_settings,pert); % run simulation    
    Y(end+1:end+length(n)) = popfuncs.find_arrhythmias(chunk_settings,X); % Separate cells based on susceptiblity
    
    if variations > 1000
        matfile = fullfile(TrigFolder, ['Trigger_' num2str(ii) '.mat']);
        save(matfile, 'X')
    end
    disp([num2str((ii/size(intervals,1))*100) '% Finished '])
end

A = sum(Y);
B = length(Y) - A;

figure
bar(1,B,0.5,'FaceColor',[0, 0.4470, 0.7410])
hold on
bar(2,A,0.5,'FaceColor',[0.6350, 0.0780, 0.1840])
xticks(1:2)
xticklabels({'(-)arrhythmia','(+)arrhythmia'})
ylabel('Count','FontSize',14,'FontWeight','bold','FontName','Calibri')
set(gca,'FontSize',14,'FontWeight','bold','FontName','Calibri','XGrid','on','YGrid','on')
ylim([0 length(Y)])
picfile = fullfile(yourFolder, 'TriggerBarPlot.fig');
saveas(gcf,picfile)
close(gcf)

% Save Data
matfile = fullfile(yourFolder, 'Y.mat');
Y = logical(Y)';
save(matfile, 'Y')
    
%% -- Nested Functions 
    function intervals = chunks(variations)
    if variations > 1000
        ints = round(linspace(0,variations,variations/500));
        intervals = zeros((length(ints) - 1),2);
        for i = 1:(length(ints) - 1)
            intervals(i,1:2) = [ints(i)+1 ints(i+1)];
        end

        % Create a Folder to store all of the population data.
        TrigFolder = [SubFolder '\Population'];
        mkdir(TrigFolder);
    else
        intervals = [1 variations];
    end