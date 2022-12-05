% reservoir computing sandbox
% Jason D. Bakos

close all;

% number of inputs
M = 20;
% number of reservoir nodes
N = 10;
% number of outputs
Q = 20;

% randomly initialize weights
W_in = randn(N,M);
W = randn(N,N);
W_out = randn(N,Q);

% inputs
signal = L96;

% prepare input signal
orig_samples = size(signal,1)
obs = floor(orig_samples/M)
signal = signal(1:(obs*M),1);
U = reshape(signal,M,obs)';

for epoch=1:5

    % find state sequence
    X = zeros(obs,N);
    Y = zeros(obs,Q);
    for i=2:size(U,1)
        % (N x M) * (M,1) + (N x N) * (N x 1)
        X(i,:) = tanh(W_in*U(i-1,:)' + W*X(i-1,:)')';
        % output
        Y(i-1,:) = X(i,:) * W_out;
    end

    figure;
    plot(U(:,1));
    hold on;
    plot(Y(:,1));
    title('before training');
    legend({'input','output'});
    rmse = mean((U(:,1)-Y(:,1)).^2)^.5

    % train output layer
    W_out = inv(X' * X) * X'*Y;

end

figure;
plot(U(:,1));
hold on;
plot(Y(:,1));
title('after training');
legend({'input','output'});
rmse_after = mean((U(:,1)-Y(:,1)).^2)^.5

function L96 = L96
    N = 5; % variables
    F = 8; % forcing
    t = 0:0.01:30;
    signal = zeros(numel(t),N);
    signal(1,:) = F*ones(1,N);
    signal(1,1) = signal(1,1) + 0.1;
    d = zeros(numel(t),N);
    
    n=2;
    for x = t
        for i = 0:N-1
            d = signal(n-1,mod(i+1,N)+1);
            d = d - signal(n-1,mod(i-2,N)+1);
            d = d * signal(n-1,mod(i-1,N)+1);
            d = d - signal(n-1,i+1);
            d = d + F;
            signal(n,i+1) = signal(n-1,i+1) + d*(t(2)-t(1));
        end
        n = n + 1;
    end
    
    figure;
    plot3(signal(:,1),signal(:,2),signal(:,3));
    xlabel("$x_1$",'interpreter','latex');
    ylabel("$x_2$",'interpreter','latex');
    zlabel("$x_3$",'interpreter','latex');
    
    L96 = signal;
end
