%% Figure 2C-D:
% colors 
blue = [0, 0.4470, 0.7410];
red = [0.6350, 0.0780, 0.1840];

%% Load Data
features = readtable('Data\Population_Feature_Outputs.csv');
test_index = readmatrix('Data\1Hz_14Features_GKr0.06.xlsx','Sheet','Prediction');
test_index = test_index(:,1);

APD90 = features.APD90;
APD90_test = APD90(test_index);
%% APD for IKr Block 
resp = features.IKrBlock_Label;
[AUC_IKr,FPR_IKr,TPR_IKr] = plotROC(APD90_test,resp(test_index));

label = logical(resp);
n = 10;
resistant = APD90(~label);
temp = min(resistant):n:max(resistant);
bins = linspace(min(resistant),max(resistant),length(temp));

figure
subplot(1,3,1)
histogram(resistant,bins,'DisplayStyle','stairs','linewidth',2,'EdgeColor',blue)
hold on

susceptible = APD90(label);
temp = min(susceptible):n:max(susceptible);
bins = linspace(min(susceptible),max(susceptible),length(temp));

histogram(susceptible,bins,'DisplayStyle','stairs','linewidth',2,'EdgeColor',red)
title('IKr Block')
xlabel('APD90')
ylabel('Count')
set(gca,'FontSize',14,'FontWeight','bold','FontName','Calibri')
%% APD for ICaL Pert
resp = features.ICaLIncrease_Label;
[AUC_ICaL,FPR_ICaL,TPR_ICaL] = plotROC(APD90_test,resp(test_index(:,1)));

label = logical(resp);
n = 10;
resistant = APD90(~label);
temp = min(resistant):n:max(resistant);
bins = linspace(min(resistant),max(resistant),length(temp));

subplot(1,3,2)
histogram(resistant,bins,'DisplayStyle','stairs','linewidth',2,'EdgeColor',blue)
hold on

susceptible = APD90(label);
temp = min(susceptible):n:max(susceptible);
bins = linspace(min(susceptible),max(susceptible),length(temp));

histogram(susceptible,bins,'DisplayStyle','stairs','linewidth',2,'EdgeColor',red)
title('ICaL Increase')
xlabel('APD90')
ylabel('Count')
set(gca,'FontSize',14,'FontWeight','bold','FontName','Calibri')
%% APD for Current Inject 
resp = features.Inject_Label;
[AUC_Inject,FPR_Inject,TPR_Inject] = plotROC(APD90_test,resp(test_index(:,1)));

label = logical(resp);
n = 10;
resistant = APD90(~label);
temp = min(resistant):n:max(resistant);
bins = linspace(min(resistant),max(resistant),length(temp));

subplot(1,3,3)
histogram(resistant,bins,'DisplayStyle','stairs','linewidth',2,'EdgeColor',blue)
hold on

susceptible = APD90(label);
temp = min(susceptible):n:max(susceptible);
bins = linspace(min(susceptible),max(susceptible),length(temp));

histogram(susceptible,bins,'DisplayStyle','stairs','linewidth',2,'EdgeColor',red)
title('Current Injection')
xlabel('APD90')
ylabel('Count')
set(gca,'FontSize',14,'FontWeight','bold','FontName','Calibri')
set(gcf,'Position',[472 669.6667 971 230.3333])
%% Figure 2D 
figure
subplot(1,2,1)
plot(FPR_IKr,TPR_IKr,'linewidth',2)
hold on
plot(FPR_ICaL,TPR_ICaL,'linewidth',2)
plot(FPR_Inject,TPR_Inject,'linewidth',2)
xlabel('FPR')
ylabel('TPR')
plot([0,1],[0,1],'k:','LineWidth',1.5)
set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri','XGrid','On','YGrid','On')

subplot(1,2,2)
bar(1,AUC_IKr)
hold on
bar(2,AUC_ICaL)
bar(3,AUC_Inject)
ylabel('AUC')
ylim([0.5 1])
xticks(1:3)
xticklabels({'IKr Block','ICaL Increase','Current Inject'})
xtickangle(90)
set(gca,'FontSize',12,'FontWeight','bold','FontName','Calibri','XGrid','On','YGrid','On')
set(gcf,'Position',[472 480 970.3333 420])