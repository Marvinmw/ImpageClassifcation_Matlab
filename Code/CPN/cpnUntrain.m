%Kohonen Algorithm
function [ cpn ] = cpnUntrain( x,cpn )
%CPNTRAIN Summary of this function goes here
%   Detailed explanation goes here
numsample=size(x,1);
tmax=numsample*2;
iR=mod(randperm(tmax),numsample)+1;
sumall=zeros(size(cpn.hidden,1));
average=zeros(size(cpn.hidden,1));
for t=1:tmax
   i=iR(t);
  xnorm=x(i,:)./norm(x(i,:));
 [nextcpn]=som_step(cpn,xnorm);
    
     %%statistics 
     
     for p=1:size(cpn.hidden,1)
     sumall(p)=sumall(p)+abs(norm(nextcpn.hidden(p,:)-cpn.hidden(p,:),2));
     average(p,t)=sumall(p)/t;
     end 
    ave=mean(average);
    %%converngence criteria
    if(t>=2)
        if(t>numsample && abs(ave(t)-ave(t-1))< 1e-3)
            break;
        end
    end
    cpn=nextcpn;  
    %%decrease sigma with the time
     if(mod(t,100)==0)
         cpn.learningrate=0.8*cpn.learningrate;
     end
     if(cpn.learningrate<1e-5)
         break;
     end
end
end

