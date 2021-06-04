function output = mypredict2(mynet,x_input)

    weights_hidden = mynet.weights_hidden;
    bias_hidden = mynet.bias_hidden;
    weights_output = mynet.weights_output;
    bias_output = mynet.bias_output;

    wl = x_input.WordLength;
    fl = x_input.FractionLength;

    out_hidden = fi(weights_hidden * x_input,1,wl,fl);
    out_hidden = fi(out_hidden + bias_hidden,1,wl,fl);
    output = fi(weights_output * out_hidden,1,wl,fl);
    output = fi(output + bias_output,1,wl,fl);

end
