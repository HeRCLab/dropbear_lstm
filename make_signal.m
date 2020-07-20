function [x,signal] = make_signal (sample_rate,time,amps,freqs,phases)

    points = sample_rate * time;
    x = single(time/points * [0:(points-1)]); % for plotting

    signal = single(zeros(1,size(x,2)));
    
    for i=1:size(amps,2)
        amplitude = amps(i);
        frequency = freqs(i);
        phase = phases(i);
        
        t = x * 2*pi * frequency + phase;
        signal = signal + amplitude * sin(t);
    end
    
end