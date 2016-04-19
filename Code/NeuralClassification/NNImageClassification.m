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
[rdata,ry]=randomData(train.X_cnn,train.y);
N=size(rdata,1);
Tr=[];
Tr.index=1:(0.8*N);
Tr.X=rdata(Tr.index,:);
Tr.y=ry(Tr.index);

Te=[];
Te.index=(length(Tr.index)+1):N;
Te.X=rdata(Te.index,:);
Te.y=ry(Te.index);
N=size(Tr.X,2);

%set up the backforward network neuron
%netarchitectures={[N 50  4],[N 100  4],[N 150  4],[N 200  4],[N 300  4],[N 100 50 4],[N 200 50 4],[N 300 200 100 50 4]};
%netarchitectures={[N 4],[N 100  4],[N 200 100 4],[N 400 200 100 4],[N 500 400 200 100 4],[N 500 400 200 100 50 4],[N 500 400 200 100 50 4]};
%netarchitectures={[N 200  4],[N 200 150 100 50 40 20 10 4]};

netarchitectures={[N  600 300 4]};
for i=1:length(netarchitectures);
nn=nnsetup(netarchitectures{1});

nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate                     = 1;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
nn.momentum                         = 0.5;          %  Momentum
nn.scaling_learningRate             = 0;            %  Scaling factor for the learning rate (each epoch)
nn.weightPenaltyL2                  = 0;            %  L2 regularization
nn.nonSparsityPenalty               = 0;          %  Non sparsity penalty
nn.sparsityTarget                   = 0;         %  Sparsity target
nn.inputZeroMaskedFraction          = 0.5;            %  Used for Denoising AutoEncoders
nn.dropoutFraction                  = 0.5;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
opts.numepochs=15;
opts.batchsize=20;
opts.validation =0;
%if ==1 => plots training error as the NN is trained
opts.plot=0;
%choose the data to satisfy the form of the program
numSampToUse=opts.batchsize*floor(size(Tr.X)/opts.batchsize);
Tr.X=Tr.X(1:numSampToUse,:);
Tr.y=Tr.y(1:numSampToUse);

%Train Label
LL=[1*(Tr.y==1),1*(Tr.y==2),1*(Tr.y==3),1*(Tr.y==4)];


%Test Label 
LLe=[1*(Te.y==1),1*(Te.y==2),1*(Te.y==3),1*(Te.y==4)];

[nn,L]=nntrain(nn,Tr.X,LL,opts);
%predict
prelabels = nnpredict(nn, Te.X);
prelabels = [1*(prelabels==1),1*(prelabels==2),1*(prelabels==3),1*(prelabels==4)];

trainlable = nnpredict(nn, Tr.X);
trainlable = [1*(trainlable==1),1*(trainlable==2),1*(trainlable==3),1*(trainlable==4)];

%compute the balance error rate
[ber(i) table]=balanceErrorRate(prelabels,LLe);
[tber(i) ~]=balanceErrorRate(trainlable,LL);
disp(['test',num2str(ber(i)),' train',num2str(tber(i))]);

end


%binary classification
bprelabels = nnpredict(nn, Te.X);
bpre=bprelabels<4;
bpre=[1*(bpre==0),1*(bpre==1)];
%test 
Te.by=Te.y<4;
bLLe=[1*(Te.by==0),1*(Te.by==1)];
[bber, btable]=balanceErrorRate(bpre,bLLe);
load test.mat
multiplepre=nnpredict(nn,test.X_cnn);
binarypre=multiplepre<4;


