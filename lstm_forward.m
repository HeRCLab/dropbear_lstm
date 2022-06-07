function [cell_state,hidden_state] = lstm_forward(cell_state,hidden_state,layer,x)

    perms = [1 0 3 2];

    % (units * 4) x 1
    input_weights = layer.InputWeights;

    % (units * 4) x units
    recurrent_weights = layer.RecurrentWeights;

    % (units * 4) x 1
    bias = layer.Bias;

    % infer the number of units (also in
    num_units = size(cell_state,1);

    % allocate output signal
    signal_pred = zeros(1,num_units);

    % get values for current gate (since they are packed)
    segment = perms(1);
    chunk = num_units*segment+1:num_units*(segment+1);
    %chunk = 1:4:num_units*4;
    % (units x 1) x (1 x 1) + (units x units) x (units x 1) + (units x 1)

    forget_gate = input_weights(chunk,:) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
    forget_gate = sigmoid(forget_gate);

    segment = perms(2);
    chunk = num_units*segment+1:num_units*(segment+1);
    %chunk = 2:4:num_units*4;
    input_gate = input_weights(chunk,:) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
    input_gate = sigmoid(input_gate);

    segment = perms(3);
    chunk = num_units*segment+1:num_units*(segment+1);
    %chunk = 3:4:num_units*4;
    output_gate = input_weights(chunk,:) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
    output_gate = sigmoid(output_gate);

    segment = perms(4);
    chunk = num_units*segment+1:num_units*(segment+1);
    %chunk = 4:4:num_units*4;
    cell_state_next = input_weights(chunk,:) * x + recurrent_weights(chunk,:) * hidden_state + bias(chunk,1);
    cell_state_next = tanh(cell_state_next);

    cell_state = forget_gate .* cell_state + input_gate .* cell_state_next;

    hidden_state = output_gate .* tanh(cell_state);
   
end
