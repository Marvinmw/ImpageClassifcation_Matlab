function [ber,table]=balanceErrorRate(predic,labels)
% compute the table of the prediction and the  truth
  for i=1:size(predic,2)
      for j=1:size(labels,2)
      temp=predic(:,i)+labels(:,j);
      table(i,j)=length(find(temp==2)); 
      end
  end
%compute the ber
      rightNoclusters=sum(table,1)';
      preCrrectNo=diag(table);
      ber=1.0-mean(preCrrectNo./rightNoclusters);  
      if(isnan(ber))
      table
      end
end