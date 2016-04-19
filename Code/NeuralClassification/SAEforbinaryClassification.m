
clear all
load('train/train.mat')
%product test and train data
temp=clock;
temp=sum(temp(4:6))*sum(temp(2:3));
temp=round(temp/10);
rand('seed',temp);
%randomly select the train data and the test data
[trainX,trainY]=randomData(train.X_cnn,train.y);
Tr.index=1:(0.8*size(trainX,1));
Tr.X=trainX(Tr.index,:);
Tr.y=trainY(Tr.index)<4;

Te.index=(length(Tr.index)+1):size(trainX,1);
Te.X=trainX(Te.index,:);
Te.y=trainY(Te.index)<4;

xnum=size(trainX,2);
sizenetwork=[xnum   400 ];
sae=saesetup(sizenetwork);

numnetworks=length(sae.ae);
%initial network

   sae.ae{1}.activation_function       = 'sigm';
   sae.ae{1}.learningRate              = 1;
   sae.ae{1}.momentum                         =0;          %  Momentum
   sae.ae{1}.weightPenaltyL2                  = 0;            %  L2 regularization
   sae.ae{1}.nonSparsityPenalty               = 0;          %  Non sparsity penalty
   sae.ae{1}.sparsityTarget                   = 0.05;         %  Sparsity target
   sae.ae{1}.output                           = 'sigm';  
   sae.ae{1}.inputZeroMaskedFraction   = 0.8;
   sae.ae{1}.dropoutFraction=0;
   
   opts.numepochs=10;
   opts.batchsize=25;
   opts.plot=0;   
%choose the data to satisfy the form of the program
numSampToUse=opts.batchsize*floor(size(Tr.X)/opts.batchsize);
Tr.X=Tr.X(1:numSampToUse,:);
Tr.y=Tr.y(1:numSampToUse);
% normalize data
%[Tr.normX, mu, sigma] = zscore(Tr.uX); % train, get mu and std

sae = saetrain(sae,Tr.X, opts);

%initiate the a network NN not include the out layer[xnum 500 100]
nn=nnsetup([xnum   400 2]);
nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate                     = 1;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
nn.momentum                         = 0;          %  Momentum
nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
nn.weightPenaltyL2                  = 0;            %  L2 regularization
nn.nonSparsityPenalty               = 0;          %  Non sparsity penalty
nn.sparsityTarget                   = 0;         %  Sparsity target
nn.inputZeroMaskedFraction          = 0.5;            %  Used for Denoising AutoEncoders
nn.dropoutFraction                  = 0.5;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
nn.W{1}=sae.ae{1}.W{1};

%train the newnetwork
LLe=[1*(Te.y==0),1*(Te.y==1)];
LL=[1*(Tr.y==0),1*(Tr.y==1)];
%using the NN network ouput
nn = nntrain(nn, Tr.X, LL, opts);

 %classify
prelabels=nnpredict(nn, Te.X);
prelabels = [1*(prelabels==0),1*(prelabels==1)];
[ber table]=balanceErrorRate(prelabels,LLe);
disp(ber);




