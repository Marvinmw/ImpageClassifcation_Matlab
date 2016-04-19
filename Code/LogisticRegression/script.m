clear all
clearvars;

% Load features and labels of training data
load train/train.mat;
addpath(genpath('DeepLearnToolbox-master'));
temp=clock;
temp=sum(temp(4:6))*sum(temp(2:3));
temp=round(temp/10);
rand('seed',temp);
%Train Data and Test Data
Tr=[];
[trainX,trainY]=randomData(train.X_cnn,train.y);
N=size(trainX,1);
Tr.index=1:(0.8*N);
Tr.X=trainX(Tr.index,:);
Tr.y=trainY(Tr.index);
[Tr.normx,mu,sigma]=zscore(Tr.X);
Tr.by=Tr.y<4;

Te=[];
Te.index=length(Tr.index):size(trainX,1);
Te.X=trainX(Te.index,:);
Te.y=trainY(Te.index);
[Te.normx,mu,sigma]=zscore(Te.X);
Te.by=Te.y<4;

X=[ones(size(Tr.normx,1),1)  Tr.normx];
%learning rate.
alpha=2;

[ beta , ltt ,lte, rt ,re,rmset,rmsee] = KCVLogisticNoP(X,Tr.by,alpha,10,1);
ex=[ones(size(Te.normx,1),1)  Te.normx];
pre=sigmoid(ex*beta);
pre=pre>0.5;

bpre=[pre==0,pre==1];
cy=[Te.by==0,Te.by==1];
[ber,table]=balanceErrorRate(bpre,cy);






