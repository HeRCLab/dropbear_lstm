function training_accuracy = evaluate_net (weights_hidden,bias_hidden,weights_output,bias_output,x_train,y_train)

out_hidden{1}=x_train * weights_hidden{1}' + ones(size(x_train,1),1)*bias_hidden{1};
for i=2:size(weights_hidden,2)
    out_hidden{i}=out_hidden{i-1} * weights_hidden{i}' + ones(size(out_hidden{i-1},1),1)*bias_hidden{i};        % 60000x784 * 784x10 = 60000x10
end
out_output=out_hidden{size(weights_hidden,2)} * weights_output' + ones(size(out_hidden{size(weights_hidden,2)},1),1)*bias_output;   % 60000x10 * 10x10 = 60000x10

training_accuracy=(mean((out_output-y_train).^2))^.5;
