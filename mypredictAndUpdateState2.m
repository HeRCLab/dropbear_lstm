function [net,signal_pred,cell_state_sequence,hidden_state_sequence] = mypredictAndUpdateState2(net,train_x)

    hidden_states = {};
    cell_states = {};
    
    cell_state_sequence = {};
    hidden_state_sequence = {};   
    
    signal_pred=zeros(1,size(train_x,2));
    
    % allocate hidden and cell states for all LSTM layers
	for i = 1:size(net.Layers,1)
        layer = net.Layers(i);
        if strcmp(class(layer),'nnet.cnn.layer.LSTMLayer')
            hidden_states = [hidden_states zeros(size(layer.HiddenState))];
            cell_states = [cell_states zeros(size(layer.CellState))];
            cell_state_sequence = [cell_state_sequence, zeros(size(layer.CellState,1),size(train_x,2))];
            hidden_state_sequence = [hidden_state_sequence, zeros(size(layer.CellState,1),size(train_x,2))];
        elseif strcmp(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
            fully_connected_weights = layer.Weights;
            fully_connected_bias = layer.Bias;
        end
    end
    
    for i = 1:size(train_x,2)
        x = train_x(:,i);
        n = 1;
        for j = 1:size(net.Layers,1)
            layer = net.Layers(j);
            if strcmp(class(layer),'nnet.cnn.layer.LSTMLayer')
                [cell_states{n},hidden_states{n}] = lstm_forward(cell_states{n},hidden_states{n},layer,x);
                cell_state_sequence{n}(:,i) = cell_states{n};
                hidden_state_sequence{n}(:,i) = hidden_states{n};
                x = hidden_states{n};
                n = n+1;
            elseif strcmp(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
                signal_pred(1,i) = fully_connected_weights * hidden_states{n-1} + fully_connected_bias;
            end
        end
    end
end
