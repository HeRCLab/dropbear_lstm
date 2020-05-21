function [x,signal] = make_signal (sample_rate,time,amps,freqs,phases)

    points = sample_rate * time;
    x = time/points * [1:points]; % for plotting

    signal = zeros(1,size(x,2));
    
    for i=size(amps,2)
        amplitude = amps(i);
        frequency = freqs(i);
        phase = phases(i);
        
        t = x * 2*pi * frequency + phase;
        signal = signal + amplitude * sin(t);
    end

end