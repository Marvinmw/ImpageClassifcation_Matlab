function [ cpn ] = som_step( cpn,data )
%som_step   performs one step of the sequential learning for a self
%           organized map (som)
%
%  som_step(cpn,data)

sizeK = sqrt(size(cpn.hidden,1));

%find the best matching unit via the minimal distance to the datapoint
[~, winningunit]=min(sqrt(sum(abs(cpn.hidden-repmat(data,sizeK^2,1)).^2,2)));

%find coordinates of the winner
[a, b]=find(cpn.neighbour==winningunit);
disc=1;
j=winningunit;
% update winner and neighbors according to the neighborhood function    
cpn.hidden(j,:)=cpn.hidden(j,:)+disc*cpn.learningrate*(data-cpn.hidden(j,:));
cpn.hidden(j,:)=cpn.hidden(j,:)./norm(cpn.hidden(j,:));




end

