function [SSI ASI] = SpAbScIndex(psth1,psth2)
%The SSI is to measure how much the response of a neuron to differnt
%intervals can not be described by the absolute-scaling scenario
%The absolute and scaling index (ASI) is to measure the absolute (or scaling) level of
% the responses of one neuron (with low SSI) to two intervals.
% psth1: population dynamics for short interval;1-d neuron; 2-d time
% psth2: population dynamics for long interval;1-d neuron; 2-d time
% By Shanglin Zhou

maxTime1 = size(psth1,2);
maxTime2 = size(psth2,2);
numCell = size(psth1,1);

%% Absolute-scaling index (ASI)
abscDist = zeros(numCell,maxTime1-1);%Euclidean distance between psth1 and time-warped psth2
AbsR =  zeros(numCell,maxTime1-1);% absoulte ratio
for i = 1:maxTime1-1
    psth2Temp = [psth2(:,1:i) psth2(:,int16(linspace(i+1,maxTime2,maxTime1-i)))];% time-warped psth2 at break point i
    temp1 = (psth1-psth2Temp).^2;
    abscDist(:,i) = sqrt(sum(temp1,2));
    
    W_abs = mean(abs((psth1(:,1:i)-psth1(:,1)).*(psth2Temp(:,1:i)-psth2Temp(:,1))),2);
    W_scale = mean(abs((psth1(:,i+1:end)-psth1(:,i+1)).*(psth2Temp(:,i+1:end)-psth2Temp(:,i+1))),2);
    AbsR(:,i) = W_abs./(W_abs + W_scale);
    
    [r,p] = corr(psth1',psth2Temp');
    CORR(:,i) = diag(r)';
    P(:,i) = diag(p)';
end

AbsR(isnan(AbsR)) = 0.5;

[~,minInd] = min(abscDist,[],2);% minimun distance between psth1 and time-warped psth2 at minInd
temporal_Index = minInd./maxTime1;

AbsR_Index = AbsR(sub2ind(size(AbsR),1:numCell,minInd'));

ASI = (temporal_Index + AbsR_Index')/2;

%% stimulus-specific index (SSI)
SSI = 1 - CORR(sub2ind(size(AbsR),1:numCell,minInd'))';
end