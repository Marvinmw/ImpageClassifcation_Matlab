%use the crossing validation to compute the logistic method without the
%penalized item
function [ meantheta ] = KCVSoftmax( SampleX,SampleY,alpha,Ks ,nucluster)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
rng(Ks);
N = size(SampleY,1);
idx = randperm(N);
Nk = floor(N/Ks);
for k = 1:Ks
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end
alltheta=cell(1,Ks);
for k = 1:5
		% get k'th subgroup in test, others in train
		idxTe = idxCV(k,:);
		idxTr = idxCV([1:k-1 k+1:end],:);
		idxTr = idxTr(:);
		yTe = SampleY(idxTe);
		XTe = SampleX(idxTe,:);
		yTr = SampleY(idxTr);
		XTr = SampleX(idxTr,:);
		
		% mutlti logistic
        disp(k);
        [theta] = softmaxregresssion(yTr,XTr,alpha,nucluster);
        alltheta{k}=theta;

        
        pre=getprobability(theta,XTe);
        [~,preV]=max(pre,[],2);
        predic=[1*(preV==1),1*(preV==2),1*(preV==3),1*(preV==4)];
        LLe=[1*(yTe==1),1*(yTe==2),1*(yTe==3),1*(yTe==4)];
        %compute the balance error rate
        [ber, ~]=balanceErrorRate(predic,LLe);
        disp(ber);
        
end

 meantheta=(alltheta{1}+alltheta{2}+alltheta{3}+alltheta{4})./4;
   
end





