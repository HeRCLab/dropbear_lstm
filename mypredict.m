function output = mypredict(mynet,x_input)

    weights_hidden = mynet.weights_hidden;
    bias_hidden = mynet.bias_hidden;
    weights_output = mynet.weights_output;
    bias_output = mynet.bias_output;

    out_hidden{1}=x_input * weights_hidden{1}' + ones(size(x_input,1),1)*bias_hidden{1};
    for i=2:size(weights_hidden,2)
        out_hidden{i}=out_hidden{i-1} * weights_hidden{i}' + ones(size(out_hidden{i-1},1),1)*bias_hidden{i};
    end
    
    output=out_hidden{size(weights_hidden,2)} * weights_output' + ones(size(out_hidden{size(weights_hidden,2)},1),1)*bias_output;


end
