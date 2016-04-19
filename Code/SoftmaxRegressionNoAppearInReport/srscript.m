clear all
clearvars;

% Load features and labels of training data
load train/train.mat;
addpath(genpath('DeepLearnToolbox-master'));
rng(8873);
%Train Data and Test Data
Tr=[];
Tr.index=1:2:size(train.X_cnn,1);
Tr.X=train.X_cnn(Tr.index,:);
Tr.y=train.y(Tr.index);
[Tr.normx,mu,sigma]=zscore(Tr.X);

Te=[];
Te.index=2:2:size(train.X_cnn,1);
Te.X=train.X_cnn(Te.index,:);
Te.y=train.y(Te.index);
[Te.normx,mu,sigma]=zscore(Te.X);


X=[ones(size(Tr.normx,1),1)  Tr.X(:,1:4)];
alpha=0.5;


[meantheta] = KCVSoftmax ( X,Tr.y,alpha,5,4 );

 


 ex=[ones(size(Te.normx,1),1)  Te.X(:,1:4)];

pre=getprobability(meantheta,ex);

[M,preV]=max(pre,[],2);

predic=[1*(preV==1),1*(preV==2),1*(preV==3),1*(preV==4)];
LLe=[1*(Te.y==1),1*(Te.y==2),1*(Te.y==3),1*(Te.y==4)];
%compute the balance error rate
[ber table]=balanceErrorRate(predic,LLe);
disp(ber);





