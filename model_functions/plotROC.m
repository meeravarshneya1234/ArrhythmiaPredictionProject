function [AUC, FPR, TPR] = plotROC(X,Y)
[len,~] = size(X);
P = sum(Y); % number of positives (EADs)
N = len - P;% number of negatives (no EADs)

Min_threshold = min(X);
Max_threshold = max(X);
thresholds = linspace(Min_threshold,Max_threshold,1000);
for i = 1:length(thresholds)
    FP = 0;
    TP = 0;
    for ii = 1:len
        if X(ii) >= thresholds(i)
            if Y(ii) == 1
                TP = TP + 1;
            else
                FP = FP + 1;
            end
        end
    end
    FPR(i) = FP/N;
    TPR(i) = TP/P;
    
end
AUC = abs(trapz(FPR,TPR));
if AUC < 0.5
    a = FPR;
    b = TPR;
    
    FPR = b;
    TPR = a;
    AUC = 1- AUC;
    
    
end 

