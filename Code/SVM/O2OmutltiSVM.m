%extends SVM to mutil classification
%extension methond: choose two from all the types and do SVM. K types.
%K(K-1)/2 times SVM
%for every svm(type_i, type_j),if type_i wins and vote for type_i. 
%in the end, pick up the most votes.

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
trdStruc.sub(n).SVMModel=fitcsvm(trdStruc.sub(n).x,trdStruc.sub(n).y,'KernelFunction','rbf',...
    'KernelScale','auto','BoxConstraint',5);
end
disp('step1 construct');
%voting table
votTab=zeros(size(Te.X,1),typenum);
%vote for the test X
for n=1:trdStruc.numpairs
%using matlab 3fitcsvm
[label,scores] = predict(trdStruc.sub(n).SVMModel,Te.X);
temp=[(label==1)*1,(label==2)*1,(label==3)*1,(label==4)*1];
votTab=votTab+temp;
end

[M,telabel]=max(votTab,[],2);   
tetable=[1*(telabel==1),1*(telabel==2),1*(telabel==3),1*(telabel==4)];
LLe=[1*(Te.y==1),1*(Te.y==2),1*(Te.y==3),1*(Te.y==4)];
%compute the balance error rate
[ber table]=balanceErrorRate(tetable,LLe);
disp(ber);
