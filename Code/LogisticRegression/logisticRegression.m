%compute beta using logistic regression without regularization by gradient
%descent method
function [ beta] = logisticRegression(y,tX,alpha,da)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%LOGISTICREGRESSION Summary of this function goes here
%   Detailed explanation goes here
[columns]=size(tX,2);
%initial beta
beta=rand(columns,1)*2;
max_iterators=4000;
%prebeta=beta;
for i=1:max_iterators
   % for k=1:20
[costL, gradient] = logisticRegLoss(beta, y, tX);
%iterate beta
if(mod(i,200)==0 && da==1)
alpha=alpha*0.8;
end
beta=beta-alpha*gradient;

 %compute cost
  y_pr=sigmoid(tX*beta);
  [logerror(i)]=logLoss(tX,y,beta);
  [rate(i)]=lossclassify(y,y_pr);
  [rmseerror(i)]=rmseLogistic(y,y_pr);
  
 cm(i)=costL;
%stopping criteria
  if (i~=1)
      mt=gradient;
      % disp([num2str(i),'====',num2str(mt'*mt),'rate=',num2str(rate(i))]);
   if(abs(mt'*mt)<1e-5)
       break;
   end
  end
  pregradient=gradient;


end 
disp(['alpha=',num2str(alpha),'iteration =',num2str(i)]);
end
%using nowton method logistic without regularization
function [L, g] = logisticRegLoss(beta, y, tX)
%LOGISTICREGLOSS Summary of this function goes here
%   Detailed explanation goes here
%  S=zeros(size(tX,2),size(tX,1));
%  for i=1:size(tX,1)
%   S(i,i)=sigmoid(tX(i,:)*beta)*(1-sigmoid(tX(i,:)*beta));
%  end
%  H=tX'*S*tX;
g=computeGradientc(y,tX,beta);
L=computeCostc(tX,y,beta);
end
%comput logist cost  without regularization
function [ cost] = computeCostc(X,y,beta)
%COMPUTECOST Summary of this function goes here
%   Detailed explanation goes here
cost=0;
for i=1:length(y)
   xv=X(i,:)';
   cost=cost+y(i).*xv'*beta-log(1+exp(xv'*beta));
end
cost=-cost;
end
%compute logistic gradient without regularization
function [ g ] = computeGradientc(y,X,beta)
%COMPUTEGRADIENT. Summary of this function goes here
%   Detailed explanation goes here
t=X*beta;
g=X'*(sigmoid(t)- y);
end

%compute logLoss of logistic
%comput logist cost  without regularization
function [ cost] = logLoss(X,y,beta)
%COMPUTECOST Summary of this function goes here
%   Detailed explanation goes here
cost=0;
for i=1:length(y)
   xv=X(i,:)';
   if(isinf(exp(xv'*beta)))
       cost=cost+y(i).*xv'*beta-xv'*beta;
   else
   cost=cost+y(i).*xv'*beta-log(1+exp(xv'*beta));
   end
end
cost=-cost/length(y);
end
%compute how many the proportation of the wrong y_prs 
function [rate]=lossclassify(y_tr,y_pr)
number=0;
for i=1:length(y_tr)
    if((y_pr(i)>=0.5 && y_tr(i)==1) || (y_pr(i)<0.5 && y_tr(i)==0))
        number=number+1;
    end
end
rate=1-number./length(y_tr);
end
%compute rmse of logistic 
function [error]=rmseLogistic(y_tr,y_pre)
e=y_tr-y_pre;
error=(e'*e)/length(y_tr);
end
%logistic function
function g = sigmoid(x)  
g = exp(x) ./ (1.0 + exp(x));  
for i=1:length(g)
if(isnan(g(i)))
    g(i)=1;
end
end
end