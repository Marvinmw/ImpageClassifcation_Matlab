%use the crossing validation to compute the logistic method without the
%penalized item
function [ beta , ltt ,lte, rt ,re,rmset,rmsee] = KCVLogisticNoP( SampleX,SampleY,alpha,Ks )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
rng(Ks);
N = size(SampleY,1);
idx = randperm(N);
Nk = floor(N/Ks);
for k = 1:Ks
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for k = 1:Ks
		% get k'th subgroup in test, others in train
		idxTe = idxCV(k,:);
		idxTr = idxCV([1:k-1 k+1:end],:);
		idxTr = idxTr(:);
		yTe = SampleY(idxTe);
		XTe = SampleX(idxTe,:);
		yTr = SampleY(idxTr);
		XTr = SampleX(idxTr,:);
		
		% logistic
        [ abeta(k,:) ] = logisticRegression(yTr,XTr,alpha);
          
		
         %compute cost for training and test
  y_pr=sigmoid(XTr*abeta(k,:)');
  [logerror(k)]=logLoss(XTr,yTr,abeta(k,:)');
  [rate(k)]=lossclassify(yTr,y_pr);
  [rmseerror(k)]=rmseLogistic(yTr,y_pr);
    y_pre=sigmoid(XTe*abeta(k,:)');
  [logerrore(k)]=logLoss(XTe,yTe,abeta(k,:)');
  [ratee(k)]=lossclassify(yTe,y_pre);
  [rmseerrore(k)]=rmseLogistic(yTe,y_pre);
		
	 disp(['rate ',num2str(rate(k)),num2str(ratee(k)),'logerror',num2str(logerror(k)),num2str(logerrore(k))]);

end
   %return the result
     ltt=logerror;
     lte=logerrore;
     rt=rate;
     re=ratee;
     rmset=rmseerror;
     rmsee=rmseerrore; 
     beta=mean(abeta)';
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
