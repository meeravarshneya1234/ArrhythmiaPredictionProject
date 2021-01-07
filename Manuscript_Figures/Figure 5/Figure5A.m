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
set(gcf,'Position',[314.3333 289.6667 876.6667 314])
subplot(1,2,1)
plot(t,V,'linewidth',2)
hold on
subplot(1,2,2)
plot(t,Cai,'linewidth',2)
hold on

% Run cells with high (2x) and low (0.5x) extracellular calcium 
settings.nBeats = 100;
Cao = [2 0.5];
for i = 1:2
    pert = settings_blockcurrents;
    pert.Cao = Cao(i);
    datatable = runSim(settings,pert);
    
    t = datatable.times;
    V = datatable.states(:,1);
    Cai = datatable.states(:,6);
    
    subplot(1,2,1)
    plot(t,V,'linewidth',2)

    subplot(1,2,2)
    plot(t,Cai,'linewidth',2)
    
end
subplot(1,2,1)
xlabel('time (ms)','FontSize',12,'FontWeight','bold','FontName','Calibri')
ylabel('Voltage (mV)','FontSize',12,'FontWeight','bold','FontName','Calibri')
legend ('Steady State','High Cao','Low Cao')
set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')

subplot(1,2,2)
xlabel('time (ms)','FontSize',12,'FontWeight','bold','FontName','Calibri')
ylabel('Cai (mM)','FontSize',12,'FontWeight','bold','FontName','Calibri')
set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')