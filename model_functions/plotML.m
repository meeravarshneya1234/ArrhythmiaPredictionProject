function plotML(datasets,x_labels,feature)
Folder = 'Manuscript_Data\';
index = readmatrix([Folder datasets{1} '.xlsx'],'Sheet','Prediction');

% create ROC for APD 
if ~exist('feature','var')
    save_algs = {};
    AUCs = [];
    ROCs = [];
else
    [AUC_APD90, roc_APD90(:,1), roc_APD90(:,2)] = plotROC(feature.data(index(:,1)),feature.label(index(:,1)));
    ROCs{1} = roc_APD90;
    AUCs(1) = AUC_APD90;
    save_algs{1} = 'LR';
end

for i = 1:length(datasets)
    
    [data,datalabels] = xlsread([Folder datasets{i} '.xlsx'],1);
    algs = datalabels(2:end,1);
    
    % get the index for the best alg, highest AUC 
    [~,maxI] = max(data(:,end));
    disp(['Best Alg: ' algs{maxI,1}])
    save_algs{end+1} = algs{maxI,1};
    
    AUCs(end+1) = data(maxI,end);
    
    [roc,~] = xlsread([Folder datasets{i} '.xlsx'],2);
    ROCs{end+1} = roc(:,maxI*2-1:maxI*2);
    
end    % first plot the results of all the algorithms
figure
subplot(1,2,1)
for i = 1:length(AUCs)
    b = bar(i,(AUCs(i)),1);
%     set(b,'Facecolor',colors(i,:))
    hold on
end
plot(1:(length(AUCs)+0.5),repmat(AUCs(1),1,length(AUCs)),'k--','linewidth',1)
text(1:length(AUCs),round(AUCs,2),num2str(round(AUCs,2)'),'vert','bottom','horiz','center','fontsize',10); 
ylim([0.5 1])
xticks(1:length(AUCs)+0.5)
xticklabels(x_labels)
xtickangle(45)
ylabel('AUC')
set(gca,'FontSize',12,'FontWeight','bold','FontName','CalibriLight','XGrid','off','YGrid','on')

subplot(1,2,2)
for i = 1:length(AUCs)
    plot(ROCs{i}(:,1),ROCs{i}(:,2),'linewidth',2)
    hold on
end 
plot([0,1],[0,1],'k:','LineWidth',2.5)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
legend(save_algs,'Location','SouthEast')
set(gca,'FontSize',12,'FontWeight','bold','FontName','CalibriLight','XGrid','on','YGrid','on')
set(gcf,'Position',[472 480 1015 420])