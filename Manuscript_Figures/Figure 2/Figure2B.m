%% Figure 2B 
load('Manuscript_Data\BasePop') %load population data

settings.celltype = 'endo';
settings.PCL = 1000 ;  % Pacing, Interval bewteen stimuli,[ms]
settings.stim_delay = 100 ; % Time the first stimulus, [ms]
settings.stim_dur = 2 ; % Stimulus duration
settings.stim_amp = 32.2; % Stimulus amplitude
settings.numbertokeep = 2;
settings.nBeats = 2;
cells_to_test = [37,154];
settings.ICs = popICs(cells_to_test,:);
settings.scalings = popscalings(cells_to_test,:);
settings.variations = length(cells_to_test);

% Run cells without trigger 
pert = settings_blockcurrents;
datatable = runSim(settings,pert);

for i = 1:length(cells_to_test)
    t = datatable(i).times;
    V = datatable(i).states(:,1);
    
    figure(i)
    subplot(4,1,1)
    plot(t,V,'linewidth',2)
    xlabel('time (ms)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    ylabel('Voltage (mV)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    title('Pre-Trigger Cell')
    set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')
    
end

%% Apply IKr Block 
settings.numbertokeep = 2;
settings.nBeats = 100;
pert = settings_blockcurrents;
pert.GKr = 0.06;
datatable = runSim(settings,pert);

for i = 1:length(cells_to_test)
    t = datatable(i).times;
    V = datatable(i).states(:,1);
    
    figure(i)
    subplot(4,1,2)
    plot(t,V,'linewidth',2)
    xlabel('time (ms)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    ylabel('Voltage (mV)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    title('IKr Block')
    set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')
    
end


%% Apply ICaL Increase
settings.numbertokeep = 2;
settings.nBeats = 100;
pert = settings_blockcurrents;
pert.PCa = 15.13;
datatable = runSim(settings,pert);

for i = 1:length(cells_to_test)
    t = datatable(i).times;
    V = datatable(i).states(:,1);
    
    figure(i)
    subplot(4,1,3)
    plot(t,V,'linewidth',2)
    xlabel('time (ms)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    ylabel('Voltage (mV)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    title('ICaL Increase')
    set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')
    
end

%% Apply Current Injection
settings.numbertokeep = 2;
settings.nBeats = 100;
pert = settings_blockcurrents;
pert.Inject = 0.7;
datatable = runSim(settings,pert);

for i = 1:length(cells_to_test)
    t = datatable(i).times;
    V = datatable(i).states(:,1);
    
    figure(i)
    subplot(4,1,4)
    plot(t,V,'linewidth',2)
    xlabel('time (ms)','FontSize',10,'FontWeight','bold','FontName','Calibri')
    ylabel('Voltage (mV)','FontSize',10,'FontWeight','bold','FontName','Calibri')
    title('Current Injection')
    set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')
    
end

figure(1)
set(gcf,'Position',[561 153 264 842])
sgtitle('Cell 37')

figure(2)
set(gcf,'Position',[561 153 264 842])
sgtitle('Cell 154')