function newNet = replaceWeights(oldNet,layerID,newWeights)
%REPLACEWEIGHTS Replace layer weights of DAGNetwork
%   newNet = replaceWeights(oldNet,layerID,newWeights)
%   oldNet = the DAGnetwork you want to replace weights.
%   layerID = the layer number of which you want to replace the weights.
%   newWeights = the matrix with the replacement weights. This should be
%   the original weights size.

% Split up layers and connections
oldnet = SeriesNetwork(oldNet.Layers);

% Set new weights
layers(layerID).RecurrentWeights = newWeights;

% Freeze weights, from the Matlab transfer learning example
for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

% Build new lgraph, from the Matlab transfer learning example
newLgraph = layerGraph();
for i = 1:numel(layers)
    newLgraph = addLayers(newLgraph,layers(i));
end
for c = 1:size(connections,1)
    newLgraph = connectLayers(newLgraph,connections.Source{c},connections.Destination{c});
end

% Very basic options
options = trainingOptions('sgdm','MaxEpochs', 1);

% Note that you might need to change the label here depending on your
% network in my case '1' is a valid label.
newNet = trainNetwork(zeros(oldNet.Layers(1).InputSize),1,newLgraph,options);

end
