%softmax function
%renturn a column vector
function [ gradient ] = sotmaxg( theta,x,y,k )
%SOTMAXPROBABILITY Summary of this function goes here
%   Detailed explanation goes here
    gradient=zeros(size(x,2),1);
     for i=1:size(x,1)
      p=probabilityk(theta,k,x(i,:)');
      bin=(y(i)==k);
      m=x(i,:)'.*(bin-log(p));
      gradient=gradient+m;
      gradient=gradient./size(x,1);
     end
end

function [p]=probabilityk(theta,k,x)
        totalp=0;
        v=theta'*x;
        d=max(v);
        for j=1:size(theta,2)
          totalp=totalp+exp(theta(:,j)'*x-d);
        end
        p=exp(theta(:,k)'*x-d)/totalp;
        if(isinf(totalp))
            p=0;
        end
        
end




