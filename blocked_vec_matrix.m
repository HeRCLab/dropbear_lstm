% user parameter
n = 256;

% hardware parameter
hardware_width = 64;

% initialize data
% assume we're multipling A*b, while the hardware can
% only multiply b*A, so we need to transpose A before
% mutliplying
A = rand(n,n);
b = rand(n,1);

% compute ground truth
ans1 = A * b;
ans2 = (b' * A')';

% sanity check
if ans1 ~= ans2'
    fprintf("ERROR: basic multiply test failed\n");
    return
end

% prepare the matrix for vector-matrix
A = A';
b = b';

% MMA version
ans_test = zeros(1,n);
for i=1:hardware_width:n
    for j=1:hardware_width:n
        ans_test(1,i:i+hardware_width-1) = ans_test(1,i:i+hardware_width-1) + ...
            b(1,j:j+hardware_width-1) * A(j:j+hardware_width-1,i:i+hardware_width-1);
    end
end

ans_test = ans_test';

if ans1 ~= ans_test
    fprintf("ERROR: blocked MMA multiply test failed\n");
    return
end

% vector version
ans_test = zeros(1,n);
for i=1:hardware_width:n-1
    for j=1:hardware_width:n-1
        for k=0:hardware_width-1
            ans_test(1,i+k) = ans_test(1,i+k) + ...
                b(1,j:j+hardware_width-1) * A(j:j+hardware_width-1,i+k);
        end
    end
end

ans_test = ans_test';

if ans1 ~= ans_test
    fprintf("ERROR: blocked vector multiply test failed\n");
    return
end

