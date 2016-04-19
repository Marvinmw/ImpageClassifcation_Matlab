%normalize data. Not used
function [ patches ] = normdata( patches )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
patches=bsxfun(@minus,patches,mean(patches));
pstd=3*std(patches(:));
patches=max(min(patches,pstd),-pstd)/pstd;
patches=(patches+1)*0.4+0.1;

end

