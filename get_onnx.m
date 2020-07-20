% we're building a new network

% hidden layer(s)
neurons=[10];

% build a standard network first
history_length = size(x_train,2);
layers = [imageInputLayer([history_length,1,1],'Normalization','none')];
for i=1:size(neurons,2)
    layers = [layers,fullyConnectedLayer(neurons(1,i))];
end
layers = [layers,fullyConnectedLayer(1),regressionLayer];

% randomly initilize weights
weights_hidden = cell(1,size(neurons,2));

% option 1:  random initial weights
weights_hidden{1,1} = .1 * randn(neurons(1),size(x_train,2));
layers(1,2).Weights = weights_hidden{1,1};
for i=2:size(neurons,2)
    weights_hidden{1,i} = .1 * randn(neurons(i),neurons(i-1));
    layers(1,1+i).Weights = weights_hidden{1,i};
end
weights_output = .1 * randn(size(y_train,2),neurons(size(neurons,2)));
layers(1,end-1).Weights = weights_output;

% allocate and initilize biases to 0
bias_hidden = cell(1,size(neurons,2));
for i=1:size(neurons,2)
    bias_hidden{1,i} = zeros(1,size(weights_hidden{i},1));
    layers(1,1+i).Bias = bias_hidden{1,i}';
end
bias_output = zeros(1,size(weights_output,1));
layers(1,end-1).Bias = bias_output';

% write ONYX
net = assembleNetwork(layers);
exportONNXNetwork(net,'network.onnx');