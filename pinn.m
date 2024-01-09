clear all
close all

% Define Constants
m = 1;      % mass
k = 10;      % spring-constant
b = .5;      % dampening-constant
x_0 = 1;    % initial position
v_0 = 0;    % initial velocity

% Define the modeling equation
syms t;
syms x(t);
eq = m*diff(x,t,t) + b*diff(x,t) + k*x == 0;
vel = diff(x,t);
cond = [x(0) == x_0, vel(0) == v_0];
eq = dsolve(eq, cond);

% Load the data
times = 0:.02:20;
y = zeros(size(times));

for i = 1:length(times)
    t_val = times(i);
    y(i) = subs(eq, t, t_val);
end

% Prepare the data for training
train_ratio = .5;
idx = floor(length(times) * train_ratio);

input_data = times(1:idx);
target_data = y(1:idx);

test_input = times(idx+1: end);
test_target = y(idx+1:end);

% Create the network
layers = [ ...
    sequenceInputLayer(1) %x(t)
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(1); % Pos
    regressionLayer;
];

% Set training parameters
opts = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'Verbose',1);

% Train the network
net = trainNetwork(input_data, target_data, layers, opts);
y_pred = predict(net, times);

% Evaluate the network
plot(times,y,'b-')
hold on;
plot(times,y_pred,'r-')
title('Regression w/out Physics: Expected vs Predicted')
