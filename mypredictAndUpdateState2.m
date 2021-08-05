function [net,signal_pred] = mypredictAndUpdateState2(net,train_x)

    hidden_states = {};
    cell_states = {};
    
    % allocate hidden and cell states for all LSTM layers
	for i = 1:size(net.Layers,1)
        layer = net.Layers(i);
        if strcmp(class(layer),'nnet.cnn.layer.LSTMLayer')
            hidden_states = [hidden_states zeros(size(layer.HiddenState))];
            cell_states = [cell_states zeros(size(layer.CellState))];
        elseif strcmp(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
            fully_connected_weights = layer.Weights;
            fully_connected_bias = layer.Bias;
        end
    end
    
    for i = 1:size(train_x,2)
        x = train_x(1,i);
        n = 1;
        for i = 1:size(net.Layers,1)
            layer = net.Layers(i);
            if strcmp(class(layer),'nnet.cnn.layer.LSTMLayer')
                [cell_states{n},hidden_states{n}] = lstm_forward(cell_states{n},hidden_states{n},layer,x);
                x = hidden_states{n};
                n = n+1;
            elseif strcmp(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
                signal_pred(1,i) = fully_connected_weights * hidden_state{n} + fully_connected_bias;
            end
        end
    end
end
