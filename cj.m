A = [4 1 -10 2;
    1 9 -6 5;
    -10 -6 11 -3;
    2 5 -3 1];
b = [1; 2; 3; 4];

x = rand(4,1);
r = b - A*x;
p = r;
k = 0;

while (1)
    alpha = (r' * r) / (p' * A * p);
    x = x + alpha * p;
    r_new = r - alpha * A * p;
    
    lambda = norm(r_new);
    if lambda < 1e-9
        break;
    end
    
    beta = r_new' * r_new / (r' * r);
    p = r_new + beta * p;
    r = r_new;
end