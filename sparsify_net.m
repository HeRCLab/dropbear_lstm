function net = sparsify_net (net,req_sparsity)

    recurrent_weight_rows = 0;
    
    % tally weight geometries for pre-allocation
    for i = 1:size(net.Layers,1)
        layer = net.Layers(i);
        if strcmp(class(layer),'nnet.cnn.layer.LSTMLayer')
            recurrent_weight_rows = ...
                recurrent_weight_rows + size(layer.RecurrentWeights,1);
            recurrent_weight_cols = size(layer.RecurrentWeights,2);
        end
    end
    
    % preallocate
    recurrent_weights = zeros(recurrent_weight_rows,recurrent_weight_cols);
    
    % copy weights (assume all cells are uniform)
    recurrent_weight_rows = 0;
    for i = 1:size(net.Layers,1)
        layer = net.Layers(i);
        if strcmp(class(layer),'nnet.cnn.layer.LSTMLayer')
            recurrent_weights(...
                recurrent_weight_rows + 1:...
                recurrent_weight_rows + size(layer.RecurrentWeights,1),:) = ...
                layer.RecurrentWeights;
            
            recurrent_weight_rows = ...
                recurrent_weight_rows + size(layer.RecurrentWeights,1);
        end
    end
    
    recurrent_weights_abs = abs(recurrent_weights);
    threshold_guess = max(max(recurrent_weights_abs))/2;
    threshold_step = threshold_guess/2;
    act_sparsity = sum(double(recurrent_weights_abs > threshold_guess),'all') /...
        numel(recurrent_weights_abs);
    
    while threshold_step > 1e-10
        if act_sparsity - req_sparsity < 0
            % we need to keep more weights
            threshold_guess = threshold_guess - threshold_step;
        else
            % we need to drop more weights
            threshold_guess = threshold_guess + threshold_step;
        end
        
        act_sparsity = sum(double(recurrent_weights_abs > threshold_guess),'all') /...
        numel(recurrent_weights_abs);
    
        threshold_step = threshold_step/2;
    end

    fprintf("total weights = %d, requested sparsity = %0.2f\n",...
        recurrent_weight_rows * recurrent_weight_cols,...
        req_sparsity);
    
    fprintf("threshold = %0.2f, achieved sparsity = %0.2f\n",...
        threshold_guess,...
        act_sparsity);
    
    % sparsify!
    layers = net.Layers;
    for i = 1:size(layers,1)
        if strcmp(class(layers(i)),'nnet.cnn.layer.LSTMLayer')
            mask = double(abs(layers(i).RecurrentWeights) > threshold_guess);
            layers(i).RecurrentWeights = layers(i).RecurrentWeights .* mask;
        end
    end
    net = SeriesNetwork(layers);
    
end
