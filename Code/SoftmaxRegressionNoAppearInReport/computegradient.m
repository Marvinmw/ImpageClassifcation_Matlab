function [cost, newtheta ] = computegradient(theta,x,y,alpha)
%GRADIENT Summary of this function goes here
%   Detailed explanation goes here
    
         [cost, newtheta]=softmaxCost(theta', 4, size(x,1), alpha, x', y);
     

end

