%% Figure 2A - Determine number of overlapping labels between three triggers
data = readtable('Manuscript_Data\Population_Feature_Outputs.csv');
labels = [data.IKrBlock_Label,data.ICaLIncrease_Label,data.Inject_Label];

% Determine number of overlapping labels between three triggers
sums = sum(labels,2);
similar = length(find(sums == 3)) + length(find(sums == 0));
different = size(sums,1) - similar;
figure
subplot(1,2,1)
pie([similar different])
legend('similar label','different label')

% Determine number of overlapping labels between two triggers
S = find(sums == 2 | sums == 1);
sumz = sum(labels(S,[1,2]),2);
I = length(find(sumz == 2)) + length(find(sumz == 0)); % IKr Block and ICal Increase
sumz = sum(labels(S,[1,3]),2);
II = length(find(sumz == 2)) + length(find(sumz == 0)); % IKr Block and Inject
sumz = sum(labels(S,[3,2]),2);
III = length(find(sumz == 2)) + length(find(sumz == 0)); % Inject and ICal Increase
subplot(1,2,2)
pie([I II III])
title('Similar Labels')
legend('IKr+ICaL','IKr+Inject','ICaL+Inject')