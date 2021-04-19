Files = dir(fullfile('Manuscript_Data\', '*.xlsx'));
for i = 1:length(Files)
    data = readtable([Files(i).folder '\' Files(i).name]);
    AUCs = data.AUC;
    AUCs_norm(:,i) = AUCs./max(AUCs);
end

figure
vs = violinplot(AUCs_norm(:,1:23)');
ylabel('AUCs norm','FontSize',13);
set(gca,'FontName','Calibri','FontSize',12,'GridAlpha',0.1,'GridColor',...
    [0 0 0],'XGrid','on','XTick',1:length(AUCs_norm),'XTickLabel',...
    data.Var1,'YGrid','on');