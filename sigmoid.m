function sigmoid = sigmoid (x)
  sigmoid = ones(size(x,1),size(x,2))./(ones(size(x,1),size(x,2))+exp(-x));
end
