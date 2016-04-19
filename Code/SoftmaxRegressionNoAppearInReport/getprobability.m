
%GETPROBABILITY Summary of this function goes here
%   Detailed explanation goes here

function [p]=getprobability(theta,x)
        totalp=0;
        p=[];
        for i=1:size(x,1)
           rowx=x(i,:);
           for m=1:size(theta,2)
                v=theta'*rowx';
                d=max(v);
              for j=1:size(theta,2)
                v=theta'*rowx';
                d=max(v);
                totalp=totalp+exp(theta(:,j)'*rowx'-d);
              end
              p(i,m)=exp(theta(:,m)'*rowx'-d)/totalp;
              if(isinf(totalp))
                p(i,m)=0;
              end
           end
        end
end





