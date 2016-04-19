clear all
clearvars;
% This is for the binary classification
% Load features and labels of training data
load train/train.mat;
addpath(genpath('DeepLearnToolbox-master'));
temp=clock;
temp=sum(temp(4:6))*sum(temp(2:3));
temp=round(temp/10);
rand('seed',temp);
%Train Data and Test Data
Tr=[];
N=size(train.X_cnn,1);
Tr.index=1:(0.8*N);
Tr.X=train.X_cnn(Tr.index,:);
Tr.y=train.y(Tr.index)<4;

Te=[];
Te.index=length(Tr.index):N;
Te.X=train.X_cnn(Te.index,:);
Te.y=train.y(Te.index)<4;


%set up the backforward network neuron
cpn=setupcpn([size(Tr.X,2) 4 2]);

%data dealing
[Tr.normx,mu,sigma]=zscore(Tr.X);
% the correct data lable of each cluster
LL=[1*(Tr.y==0),1*(Tr.y==1)];

%normalize test data
%data dealing
[Te.normx,mue,sigmae]=zscore(Te.X);
%test 
LLe=[1*(Te.y==0),1*(Te.y==1)];

[ cpn ] = cpnUntrain( Tr.normx,cpn );
[ cpn ] = cpnSutrain( Tr.normx,LL,cpn );

[ y ] = cpnpredict( Te.normx ,cpn);
y=y';
predic=[1*(y==0),1*(y==1)];
%compute the balance error rate
[ber table]=balanceErrorRate(predic,LLe);
disp(ber);




