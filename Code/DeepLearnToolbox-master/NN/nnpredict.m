function labels = nnpredict(nn, x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    [dummy, i] = max(nn.a{end},[],2);
    labels=i;
   % labels = [1*(i==1),1*(i==2),1*(i==3),1*(i==4)];
end