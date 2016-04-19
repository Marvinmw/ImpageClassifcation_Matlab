function [ cpn ] = setupcpn( architecture )
%SETUPCPN Summary of this function goes here
%   Detailed explanation goes here
cpn.archtecture=architecture;
cpn.learningrate=2.5;
cpn.betalearningrate=2.5;
cpn.sigma=4;
cpn.neighbour=reshape(1:cpn.archtecture(2),sqrt(cpn.archtecture(2)),sqrt(cpn.archtecture(2)));
cpn.hidden=rand(cpn.archtecture(2),architecture(1));
cpn.output=rand(cpn.archtecture(3),cpn.archtecture(2));
cpn.epoch=15;
for i=1:size(cpn.hidden,1)
    cpn.hidden(i,:)=cpn.hidden(i,:)./norm(cpn.hidden(i,:));
end

for i=1:size(cpn.output,1)
    cpn.output(i,:)=cpn.output(i,:)./norm(cpn.output(i,:));
end

 
end
