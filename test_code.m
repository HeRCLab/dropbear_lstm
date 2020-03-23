NOISE = .1;

figure;
range = 0:.001:2*pi;
for freq = 1:size(range,2)
    clf;
    signal = sin(range * freq);
    for i=1:size(signal,2)
        signal(1,i) = signal(1,i) + rand()*NOISE;
    end
    signal_f = fft(signal);
    plot(real(signal_f));
    xlim([0,size(signal,2)]);
    pause(.01);
end

return;

SIZE = 1000;
BANDS = 5;
BANDWIDTH = 100;

% allocate freq spectrum
bands = zeros(1,SIZE);

for b=1:BANDS
    % randomly select a band
    rand_idx = ceil(rand()*SIZE);   
    rand_width = rand()*BANDWIDTH;

    for i=1:SIZE
        bands(1,i) = bands(1,i) + normpdf(i-rand_idx,rand_width);
    end
end

bands = bands/sum(bands) * SIZE;
bands = bands - max(bands)/2;

subplot(3,1,1);
plot(bands);
subplot(3,1,2);
signal=ifft(bands);
plot(real(signal));
subplot(3,1,3);
plot(imag(signal));
