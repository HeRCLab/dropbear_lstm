% read Puga's signal
data=lvm_import('Ivol_Acc_Load_1S_1STD.lvm');
x = data.Segment1.data(:,1);
signal = data.Segment1.data(:,4)';
sample_rate = numel(data.Segment1.data(:,1))/data.Segment1.data(end,1);

% sweep sample period linearly from native down to 0.1 in 100 steps
mi = [];
subsample = [];
for downsample = 1 : -.01 : .01
    tau = 1/(sample_rate*downsample);
    mi_val = mutual_information(x,signal,tau);
    mi = [mi,mi_val];
    subsample = [subsample,downsample];
end

sample_rates = subsample * sample_rate;

plot(sample_rates,mi);
title('mutual information vs sample rate');
xlabel('sample rate');
ylabel('mi');
