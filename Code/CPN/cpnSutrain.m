%Supervised Algorithm
function [ cpn ] = cpnSutrain( x,y,cpn )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for i=1:cpn.epoch    
    for j=1:size(x,1)
        %find the best matching unit via the minimal distance to the datapoint
        data=x(j,:)./norm(x(j,:));
        [~, winningunit]=min(sqrt(sum(abs(cpn.hidden-repmat(data,cpn.archtecture(2),1)).^2,2)));
        cpn.output(:,winningunit)=cpn.output(:,winningunit)+cpn.betalearningrate*(y(j,:)'-cpn.output(:,winningunit));
        cpn.output(:,winningunit)=cpn.output(:,winningunit)./norm(cpn.output(:,winningunit));
        if(mod(j,100)==0)
           cpn.betalearningrate=cpn.betalearningrate*0.8; 
        end
        if(cpn.betalearningrate<1e-5)
            break;
        end
    end
end

end

