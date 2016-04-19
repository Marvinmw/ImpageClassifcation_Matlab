function [ y ] = cpnpredict( x,cpn )
%CPNPREDICT Summary of this function goes here
%   Detailed explanation goes here
sizeK = sqrt(size(cpn.hidden,1));
   for i=1:size(x,1)
       data=x(i,:);
       data=data./norm(data);
       %find the best matching unit via the minimal distance to the datapoint
       [~, winningunit]=min(sqrt(sum(abs(cpn.hidden-repmat(data,sizeK^2,1)).^2,2)));
       output=cpn.output(:,winningunit);
       [M,I]=max(output);
       y(i)=I;
   end

end

