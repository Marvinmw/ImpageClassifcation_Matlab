clear all
load('train/train.mat')
[c,s,l]=pca(train.X_cnn);
