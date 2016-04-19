%implement the directed acyclic graph-SVMs
%construct a tree and the node is SVMModel
%It will use some the same code with one-to-one SVM.
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

typenum=4;
%preprocess data and save the result to tdStruct. 
trdStruc=[];
trdStruc.numpairs=(typenum-1)*typenum/2;
n=1;
for i=1:typenum
    for j=i:typenum
      if(i==j)
        continue;
      end
    trdStruc.sub(n).y=Tr.y(Tr.y==i | Tr.y==j);
    trdStruc.sub(n).x=Tr.X(Tr.y==i | Tr.y==j,:);
    trdStruc.sub(n).type=i*10+j;
    n=n+1;
    end
end

%traning SVM and we can get typeNu*(typeNu-1)/2
for n=1:trdStruc.numpairs
%using matlab fitcsvm
trdStruc.sub(n).SVMModel=fitcsvm(trdStruc.sub(n).x,trdStruc.sub(n).y,'Standardize',true,'KernelFunction','rbf',...
    'KernelScale','auto','BoxConstraint',5);
end

%consturct the directed acyclic graph-SVM
for n=1:trdStruc.numpairs
    if(trdStruc.sub(n).type==14)
        modelstree(1).model=trdStruc.sub(n).SVMModel;%root
        modelstree(1).left=2;
        modelstree(1).right=3;
    end
    if(trdStruc.sub(n).type==24)
        modelstree(2).model=trdStruc.sub(n).SVMModel;%second level.left node
        modelstree(2).left=4;
        modelstree(2).right=5;
    end
    if(trdStruc.sub(n).type==13)
        modelstree(3).model=trdStruc.sub(n).SVMModel;%second level.right node
        modelstree(3).left=5;
        modelstree(3).right=6;
    end
    if(trdStruc.sub(n).type==34)
        modelstree(4).model=trdStruc.sub(n).SVMModel;%the third level.left node
    end
    if(trdStruc.sub(n).type==23)
        modelstree(5).model=trdStruc.sub(n).SVMModel;%the third level.medium node
    end
     if(trdStruc.sub(n).type==12)
        modelstree(6).model=trdStruc.sub(n).SVMModel;%the third level.right level
    end
end

%predict one by one not batch
Y=Te.y;
X=Te.X;
    %root
    [rlabel,scores] = predict(modelstree(1).model,X);%root
    left2X=X(rlabel~=1,:);
    y1l=Y(rlabel~=1);
    right2X=X(rlabel~=4,:);
    y1r=Y(rlabel~=4);
    % second level
    [left2Xlabel,scores] = predict(modelstree(2).model,left2X);
    [left3X]=left2X(left2Xlabel~=2,:);
    y2l=y1l(left2Xlabel~=2);
    [mediumr3X]=left2X(left2Xlabel~=4,:);
    y2mr=y1l(left2Xlabel~=4);
    [right2Xlabel,scores]= predict(modelstree(3).model,right2X);
    [mediuml3X]=right2X(right2Xlabel~=1,:);
    y2ml=y1r(right2Xlabel~=1);
    [right3X]=right2X(right2Xlabel~=3,:);
    y2r=y1r(right2Xlabel~=3);
    %third level
    [label34,scores] = predict(modelstree(4).model,left3X);
    [labelr23,scores] = predict(modelstree(5).model,mediumr3X);
    [labell23,scores] = predict(modelstree(5).model,mediuml3X);
    [label12,scores] = predict(modelstree(6).model,right3X);
    renewy=[y2l;y2mr;y2ml;y2r];
    prelabel=[label34;labelr23;labell23;label12];

renewy=[1*(renewy==1),1*(renewy==2),1*(renewy==3),1*(renewy==4)];
predicl=[1*(prelabel==1),1*(prelabel==2),1*(prelabel==3),1*(prelabel==4)];
%compute the balance error rate
[ber table]=balanceErrorRate(predicl,renewy);
disp(ber);
