function [net,signal_pred] = mypredictAndUpdateState(net,train_x)

    myperms = perms([0 1 2 3]);
    myperms = [1 0 3 2];
    for n = 1:size(myperms,1)

        myperms(n,:)
        
        % (units * 4) x 1
        input_weights = net.Layers(2,1).InputWeights;

        % (units * 4) x units
        recurrent_weights = net.Layers(2,1).RecurrentWeights;

        % (units * 4) x 1
        bias = net.Layers(2,1).Bias;

        % units x 1
        hidden_state = net.Layers(2,1).HiddenState;

        % units x 1
        cell_state = net.Layers(2,1).CellState;

        % 1 x units
        fully_connected_weights = net.Layers(3, 1).Weights;

        % 1 x 1
        fully_connected_bias = net.Layers(3, 1).Bias;

        % infer the number of units (also in
        % net.Layers(2,1).numHiddenUnits)
        num_units = size(cell_state,1);

        % allocate output signal
        signal_pred = zeros(1,num_units);
      
        for i = 1:size(train_x,2)
            x = train_x(1,i);

            % get values for current gate (since they are packed)
            segment = myperms(n,1);
            chunk = num_units*segment+1:num_units*(segment+1);
            %chunk = 1:4:num_units*4;
            % (units x 1) x (1 x 1) + (units x units) x (units x 1) + (units x 1)
            forget_gate = input_weights(chunk,1) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
            forget_gate = sigmoid(forget_gate);

            segment = myperms(n,2);
            chunk = num_units*segment+1:num_units*(segment+1);
            %chunk = 2:4:num_units*4;
            input_gate = input_weights(chunk,1) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
            input_gate = sigmoid(input_gate);

            segment = myperms(n,3);
            chunk = num_units*segment+1:num_units*(segment+1);
            %chunk = 3:4:num_units*4;
            output_gate = input_weights(chunk,1) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
            output_gate = sigmoid(output_gate);

            segment = myperms(n,4);
            chunk = num_units*segment+1:num_units*(segment+1);
            %chunk = 4:4:num_units*4;
            cell_state_next = input_weights(chunk,1) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
            cell_state_next = tanh(cell_state_next);

            cell_state = forget_gate .* cell_state + input_gate .* cell_state_next;

            hidden_state = output_gate .* tanh(cell_state);

            signal_pred(1,i) = fully_connected_weights * hidden_state + fully_connected_bias;
        end
    
%         close all;
%         plot(train_x,'b');
%         hold on;
%         plot(signal_pred,'r');
%         legend({'train\_y','signal\_pred'});
%         hold off;
%         pause;
        
    end
end
