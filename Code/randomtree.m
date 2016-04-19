
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
tb=TreeBagger(1500,Tr.X,Tr.y);


Te=[];
Te.index=(length(Tr.index)+1):dnu;
Te.X=rdata(Te.index,:);
Te.y=ry(Te.index);



%using matlab fitcsvm

[label,scores] = predict(tb,Te.X);
label=cellfun(@str2num,(label));
correctNumber=length(find(Te.y==label));
errorrate=1-correctNumber./length(Te.y);
L=label;
tetable=[1*(L==1),1*(L==2),1*(L==3),1*(L==4)];
LLe=[1*(Te.y==1),1*(Te.y==2),1*(Te.y==3),1*(Te.y==4)];
%compute the balance error rate
[ber table]=balanceErrorRate(tetable,LLe);
disp(ber);


