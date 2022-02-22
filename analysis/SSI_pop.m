function [SSI] = SSI_pop(psth1,psth2)
%The SSI is to measure how much the response of a population to differnt
%intervals can not be described by the absolute-scaling scenario
% psth1: population dynamics for short interval;1-d neuron; 2-d time
% psth2: population dynamics for long interval;1-d neuron; 2-d time
% By Shanglin Zhou
maxTime1 = size(psth1,2);
%% Reference
for t = 1:maxTime1
    x_abs =1:t;
    x_scale = (t+2):2: t+2*(maxTime1-t);
    x = [x_abs x_scale];
    X(t,:) = x;
end
%%
DistDynamics = pdist2(psth1',psth2');
[~,distInd] = min(DistDynamics,[],2);
Dist = pdist2(distInd',X);
[~, minInd] = min(Dist);

SSI = 1 -  corr(distInd,X(minInd,:)');
end