
%binary classfication
%1=[1 2 3] 0=[4]
clear all
clearvars;
load train/train.mat;
temp=clock;
temp=sum(temp(4:6))*sum(temp(2:3));
temp=round(temp/10);
rand('seed',temp);
%Train Data and Test Data
Tr=[];
[rdata,ry]=randomData(train.X_cnn,train.y);
dnu=size(train.X_cnn,1);
Tr.index=1:(dnu*0.8);
Tr.X=rdata(Tr.index,:);
Tr.y=ry(Tr.index);



Te=[];
Te.index=(length(Tr.index)+1):dnu;
Te.X=rdata(Te.index,:);
Te.y=ry(Te.index);


%set lable to 2 binaries
Tr.by=Tr.y<4;

%set lable to 2 binaries
Te.by=Te.y<4;

kernelf={'rbf','linear','polynomial'};
boxc=[0.1 0.25 0.5 1 2 3 4 5 6 7 8 9 10];
for i=1:length(boxc)
%using matlab fitcsvm copy the fite , chagnge KernelFunction to linear and
%rbf.
SVMModel=fitcsvm(Tr.X,Tr.by,'KernelFunction','polynomial','PolynomialOrder',2,...
    'KernelScale','auto','BoxConstraint',boxc(i));

[label(:,i),scores] = predict(SVMModel,Te.X);
correctNumber=length(find(Te.by==label(:,i)));
errorrate=1-correctNumber./length(Te.by);
L=label(:,i);
tetable=[1*(L==0),1*(L==1)];
LLe=[1*(Te.by==0),1*(Te.by==1)];
%compute the balance error rate
[ber table]=balanceErrorRate(tetable,LLe);
disp(ber);
end

