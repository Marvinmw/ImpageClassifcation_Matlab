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
[ rdata, ry ] = randomData( train.X_cnn,train.y );
Tr=[];
N=size(rdata,1);
Tr.index=1:(0.8*N);
Tr.X=rdata(Tr.index,:);
Tr.y=ry(Tr.index)<4;

Te=[];

Te.index=(length(Tr.index)+1):N;
Te.X=rdata(Te.index,:);
Te.y=ry(Te.index)<4;
N=size(Tr.X,2);

%set up the backforward network neuron
netarchitectures={[N  500 250 2]};
%netarchitectures={[N 2],[N 80  2],[N 80 20 2],[N 200 150 2],[N 200 150 100 2],[N 200 150 100 50 2],[N 200 150 100 50 20 2]};
%netarchitectures={[N 2],[N 100  2],[N 100 20 2],[N 200 100 50 2],[N 200 100 50 20 2],[N 500 250 100 50 2]};
for i=1:length(netarchitectures);
nn=nnsetup(netarchitectures{i});

 nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
 nn.learningRate                     =1;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
 nn.momentum                         = 0;          %  Momentum
 nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    
 nn.weightPenaltyL2                  = 0;            %  L2 regularization
 nn.nonSparsityPenalty               = 0;          %  Non sparsity penalty
 nn.sparsityTarget                   = 0;         %  Sparsity target
 nn.inputZeroMaskedFraction          = 0.6;            %  Used for Denoising AutoEncoders
 nn.dropoutFraction                  =0.6;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
 nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
 nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'


opts.numepochs=15;
opts.batchsize=100;
opts.validation =0;
%if ==1 => plots training error as the NN is trained
opts.plot=0;

%choose the data to satisfy the form of the program
numSampToUse=opts.batchsize*floor(size(Tr.X)/opts.batchsize);
Tr.X=Tr.X(1:numSampToUse,:);
Tr.y=Tr.y(1:numSampToUse);


% the correct data lable of each class
LL=[1*(Tr.y==0),1*(Tr.y==1)];


%test label of each class
LLe=[1*(Te.y==0),1*(Te.y==1)];

[nn,L]=nntrain(nn,Tr.X,LL,opts);
%predict
prelabels = nnpredict(nn, Te.X)-1;
prelabels = [1*(prelabels==0),1*(prelabels==1)];
%compute the balance error rate
[ber(i) table]=balanceErrorRate(prelabels,LLe);


tlabels = nnpredict(nn, Tr.X)-1;
tlabels = [1*(tlabels==0),1*(tlabels==1)];
[tber(i) ~]=balanceErrorRate(tlabels,LL);
disp(['test ',num2str(ber(i)),'train',num2str(tber(i))]);
end


load('test.mat')
tlabels = nnpredict(nn, test.X_cnn)-1;

