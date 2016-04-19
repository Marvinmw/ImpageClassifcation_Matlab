%logistic function
function g = sigmoid(x)  
g = exp(x) ./ (1.0 + exp(x));  
for i=1:length(g)
if(isnan(g(i)))
    g(i)=1;
end
end
end

