function [build_ann,pred] = build_ann (x_train,y_train,neurons,epochs,alpha,varargin)

global deltas_output;

if nargin==6
    % we passed in a existing network
    weights_hidden = varargin{1}.weights_hidden;
    bias_hidden = varargin{1}.bias_hidden;
    weights_output = varargin{1}.weights_output;
    bias_output = varargin{1}.bias_output;
else
    % we're building a new network
    
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
    
    % option 2:  0 weights
    %weights_hidden{1,1} = zeros(neurons(1),size(x_train,2));
    %for i=2:size(neurons,2)
    %    weights_hidden{1,i} = zeros(neurons(i),neurons(i-1));
    %end
    %weights_output = zeros(size(y_train,2),neurons(size(neurons,2)));
    
%     % option 3:  weights = .1
%     weights_hidden{1,1} = .1 * ones(neurons(1),size(x_train,2));
%     for i=2:size(neurons,2)
%         weights_hidden{1,i} = .1 * ones(neurons(i),neurons(i-1));%0 * randn(neurons(i),neurons(i-1));
%     end
%     weights_output = .1 * ones(size(y_train,2),neurons(size(neurons,2)));
        
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
    net2 = importONNXNetwork('network.onnx','OutputLayerType','regression');

end

% allocate deltas
delta_hidden = cell(1,size(neurons,2));
for i=1:size(neurons,2)
    delta_hidden{1,i} = zeros(size(weights_hidden(i)));
end
delta_output = zeros(1,size(weights_output,1));

error_progression=zeros(1,epochs);

for i=1:epochs
    
  % adjust learning rate
  if mod(i,100)==0
      alpha = alpha/5;
  end
    
  % batch size = 1
  for j=1:size(x_train,1)
    
    % FORWARD PASS
    % compute the output of the first hidden layer
    out_hidden{1}=x_train(j,:) * weights_hidden{1}' + bias_hidden{1};
    
    % compute the outputs from subsequent hidden layers
    for hl=2:size(neurons,2)
        out_hidden{hl}=out_hidden{hl-1} * weights_hidden{hl}' + bias_hidden{hl};
    end
    
    % compute the output layer
    out_output=out_hidden{size(neurons,2)} * weights_output' + bias_output;             % 1x10 * 10x10 = 1x10
    
    % BACKWARD PASS
    % compute deltas for output layer
    delta_output=out_output-y_train(j,:);
    
    % update the output weights
    for k=1:size(weights_output,1)
        weights_output(k,:) = weights_output(k,:) - alpha * delta_output(1,k) * out_hidden{size(neurons,2)};
    end
    
    % update biases for the output layer
    bias_output = bias_output - alpha*delta_output;
    
    % compute the deltas for the last hidden layer
    for k=1:size(weights_hidden{size(neurons,2)},1)
        sum=0;
        for l=1:size(weights_output,1)
            sum = sum + delta_output(1,l)*weights_output(l,k);
        end
        delta_hidden{size(neurons,2)}(1,k)=out_hidden{size(neurons,2)}(1,k)*sum;
    end
    
    % compute deltas for all but the last hidden layer
    for hl = size(neurons,2)-1:-1:1
        for k=1:size(weights_hidden{hl},1)
            sum=0;
            for l=1:size(weights_output,1)
                sum = sum + delta_hidden{hl+1}(1,l)*weights_hidden{hl+1}(l,k);
            end
            delta_hidden{hl}(1,k)=out_hidden{hl}(1,k)*sum;
        end
    end
    
    % update the weights for all but the first hidden layer
    for hl = 2:size(neurons,2)
        for k=1:size(weights_hidden{hl},1)
            weights_hidden{hl}(k,:) = weights_hidden{hl}(k,:) - alpha * delta_hidden{hl}(1,k) * out_hidden{hl-1};
        end
    end
    
    % update the weights for the first hidden layer
    for k=1:size(weights_hidden,1)
        weights_hidden{1}(k,:) = weights_hidden{1}(k,:) - alpha * delta_hidden{1}(1,k) * x_train(j,:);
    end
    
    % update the biases for the hidden layers
    for hl=1:size(neurons,2)
        bias_hidden{hl} = bias_hidden{hl} - alpha*delta_hidden{hl};
    end
    
  end
  
  deltas_output = [deltas_output,delta_output];
  error_progression(1,i)=evaluate_net(weights_hidden,bias_hidden,weights_output,bias_output,x_train,y_train);
  
end

% figure;
% hold on;
% plot([1:epochs],error_progression,':');
% %plot([1:i],testing,'--');
% title('Training Accuracy');
% ylabel('RMSE');
% xlabel('Epoch number');
% legend('training error');

build_ann.weights_hidden = weights_hidden;
build_ann.bias_hidden = bias_hidden;
build_ann.weights_output = weights_output;
build_ann.bias_output = bias_output;

pred = out_output;
