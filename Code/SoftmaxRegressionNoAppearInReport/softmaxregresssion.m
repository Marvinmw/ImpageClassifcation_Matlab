function [ theta ] = softmaxregresssion(y,tX,alpha,clusterNu)
%SOFTMAX Summary of this function goes here
%   Detailed explanation goes here
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%LOGISTICREGRESSION Summary of this function goes here
%   Detailed explanation goes here
[columns]=size(tX,2);
%initial beta
theta=rand(columns,clusterNu)*2;
max_iterators=1000;
%prebeta=beta;
N=size(y,1);

for i=1:max_iterators
   % for k=1:20
  % disp(i);
index=randperm(N);
idx=index(1:500);
sX=tX(idx,:);  
sY=y(idx);
[cost,theta] = computegradient(theta,sX, sY,alpha);
%iterate beta
% if(mod(i,100)==0)
% alpha=alpha*0.8;
% end

disp(cost);
end 

end


