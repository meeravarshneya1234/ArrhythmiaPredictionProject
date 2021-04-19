data = readtable('Manuscript_Data\Population_Feature_Outputs.csv');

figure
subplot(3,2,1)
histogram(data.APD20)
title('APD20')

subplot(3,2,2)
histogram(data.APD50)
title('APD50')

subplot(3,2,3)
histogram(data.APD90)
title('APD90')

subplot(3,2,4)
histogram(data.CaD50)
title('CaD50')

subplot(3,2,5)
histogram(data.CaD90)
title('CaD90')

subplot(3,2,6)
histogram(data.DCai)
title('DCai')
