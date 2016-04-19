function [ rdata, ry ] = randomData( data,y )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N=size(data,1);
inx=randperm(N);
rdata=data(inx,:);
ry=y(inx,:);
end

