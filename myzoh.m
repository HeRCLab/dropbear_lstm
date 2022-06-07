function signal = myzoh (x,x_sub,signal_in)
    
    signal = zeros(1,size(x,2));
    
    i_sub = 1;
        
    for i=1:size(x,2)
        signal(1,i) = signal_in(1,i_sub);
        
        while (x(1,i) >= x_sub(1,i_sub)) && (i_sub<size(signal_in,2))
            i_sub = i_sub+1;
        end
        
        
    end

end
