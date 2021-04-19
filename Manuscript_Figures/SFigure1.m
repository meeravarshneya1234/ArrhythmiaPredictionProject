%% Load Data 
data = readtable('Manuscript_Data\Population_Feature_Outputs.csv');
label = data.Inject_Label;
test_index = readmatrix('Manuscript_Data\SteadyState_14Features_IKrBlock.xlsx','Sheet','Prediction');
test_index = test_index(:,1);
ml_rocs = readtable('Manuscript_Data\SteadyState_14Features_IKrBlock.xlsx','Sheet','ROCs');
ml_results = readtable('Manuscript_Data\SteadyState_14Features_IKrBlock.xlsx','Sheet','Results');

%% 2A
APD90 = data.APD90(test_index); % APD90 ROC curve
[AUC_APD90,FPR_APD90,TPR_APD90] = plotROC(APD90,label(test_index));
TriAP = data.TriAP(test_index); % TriAP ROC curve
[AUC_TriAP,FPR_TriAP,TPR_TriAP] = plotROC(TriAP,label(test_index));

FPR = ml_rocs.SVM_FPR;
TPR = ml_rocs.SVM_TPR;
AUC = ml_results.AUC(5);

figure
plot(FPR_APD90,TPR_APD90,'linewidth',4,'color','k')
hold on
plot(FPR_TriAP,TPR_TriAP,'linewidth',4)
plot(FPR,TPR,'linewidth',4)
xlabel('FPR')
ylabel('TPR')
plot([0,1],[0,1],'k:','LineWidth',2.5)
set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri','XGrid','On','YGrid','On')

figure
bar(1,AUC_APD90,'facecolor','k')
hold on
bar(2,AUC_TriAP)
bar(3,AUC)
ylabel('AUC')
ylim([0.5 1])
xticks(1:3)
xtickangle(90)
xticklabels({'APD90','TriAP','14 Features'})
text(1:3,round([AUC_APD90,AUC_TriAP,AUC],2),num2str(round([AUC_APD90,AUC_TriAP,AUC],2)'),'vert','bottom','horiz','center','fontsize',10); 
set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri','XGrid','On','YGrid','On')

%% 2B
load('Manuscript_Data\BasePop.mat')
cells_to_test = [792,2708];
settings.celltype = 'endo';
settings.PCL = 1000 ;  % Pacing, Interval bewteen stimuli,[ms]
settings.stim_delay = 100 ; % Time the first stimulus, [ms]
settings.stim_dur = 2 ; % Stimulus duration
settings.stim_amp = 32.2; % Stimulus amplitude
settings.numbertokeep = 1;
settings.nBeats = 1;
settings.ICs = popICs(cells_to_test,:);
settings.scalings = popscalings(cells_to_test,:);
settings.variations = length(cells_to_test);

% Run cells without trigger 
pert = settings_blockcurrents;
datatable = runSim(settings,pert);
figure
for i = 1:length(cells_to_test)
    t = datatable(i).times;
    V = datatable(i).states(:,1);
    
    plot(t,V,'linewidth',2)
    hold on
    xlabel('time (ms)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    ylabel('Voltage (mV)','FontSize',12,'FontWeight','bold','FontName','Calibri')
    set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri')
    xlim([50 750])
end
