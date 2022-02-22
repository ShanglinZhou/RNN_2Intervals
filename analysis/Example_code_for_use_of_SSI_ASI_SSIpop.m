%% Illustration of using the SSI and ASI at single cell level and SSI_pop at the populaiton level, by three prototype dynamics (scaling, absolute, stimulus-specific) as in Fig. 3 in 'Encoding time in neural dynamic regimes with distinct 1 computational tradeoffs, Zhou et al.'
% two functiona are called: SSI_pop for population level SSI and SpAbScIndex for single unit level SSI and ASI.

%% generate prototype dynamics
clc
clear
close all

seed = 5;
rand('seed',seed)
randn('seed',seed)

numUnits = 100;
maxt1 = 3000;
t1 = 1:maxt1;
maxt2 = 6000;
t2 = 1:maxt2;
width = 400;
%%% scale
x2 =zeros(numUnits, maxt2);
for n=1:numUnits
   x2(n,:) = normpdf(t2,maxt2./numUnits*n,width*2);
end
x2_scale = x2./max(x2')';

x1 = x2_scale(:,1:2:maxt2);
x1_scale = x1./max(x1')';
%%% absolute
x2 =zeros(numUnits, maxt2);
for n=1:numUnits
   x2(n,:) = normpdf(t2,maxt2./numUnits*n,width*2);
end
x2_abs = x2./max(x2')';

x1 =zeros(numUnits, maxt1);
x1(1:numUnits/2,:) = x2_scale(1:1:numUnits/2,1:maxt1);
x1_abs = x1./max(x1')';
x1_abs(isnan(x1_abs)) = 0;

%%% stimlus_specific
x2 =zeros(numUnits, maxt2);
for n=1:numUnits
   x2(n,:) = normpdf(t2,maxt2./numUnits*n,width*2);
end
x2_stim = x2./max(x2')';

x1 = x2_stim(:,1:2:maxt2);
x1_stim = x1./max(x1')';
x2_stim = x2_stim(randperm(numUnits),:);
%% plot single unit dynamics
figure(1)
selUnit = 50;

subplot(5,3,1)
plot(t1/1000,x1_scale(selUnit,:),'-','Color',[0 0 1],'LineWidth',2)
hold on
plot(t2/1000,x2_scale(selUnit,:),'Color',[0 200/255 0])
xlabel('Time (s)')
ylabel('Activity')
title('Scaling')
box off

subplot(5,3,2)
plot(t1/1000,x1_abs(selUnit,:),'-','Color',[0 0 1],'LineWidth',2)
hold on
plot(t2/1000,x2_abs(selUnit,:),'Color',[0 200/255 0])
xlabel('Time (s)')
ylabel('Activity')
title('Absolute')
box off

subplot(5,3,3)
plot(t1/1000,x1_stim(selUnit,:),'-','Color',[0 0 1],'LineWidth',2)
hold on
plot(t2/1000,x2_stim(selUnit,:),'Color',[0 200/255 0])
xlabel('Time (s)')
ylabel('Activity')
title('Stimulus\-specific')
box off 
%% plot neurogram
subplot(5,3,4)
imagesc(x1_scale)
xlabel('Time (s)')
ylabel('Units')
xticks([0 1000 2000 3000])
xticklabels({'0','1','2','3'})
% colormap('Winter')
box off

subplot(5,3,5)
imagesc(x1_abs)
xlabel('Time (s)')
ylabel('Units')
xticks([0 1000 2000 3000])
xticklabels({'0','1','2','3'})
% colormap('Winter')
box off

subplot(5,3,6)
imagesc(x1_stim(SortTraces(x1_stim),:))
xlabel('Time (s)')
ylabel('Units')
xticks([0 1000 2000 3000])
xticklabels({'0','1','2','3'})
% colormap('Winter')
box off


subplot(5,3,7)
imagesc(x2_scale)
xlabel('Time (s)')
ylabel('Units')
% colormap('Summer')
box off
xticks([0 1000 2000 3000 4000 5000 6000])
xticklabels({'0','1','2','3','4','5','6'})

subplot(5,3,8)
imagesc(x2_abs)
xlabel('Time (s)')
ylabel('Units')
% colormap('Summer')
box off
xticks([0 1000 2000 3000 4000 5000 6000])
xticklabels({'0','1','2','3','4','5','6'})

subplot(5,3,9)
imagesc(x2_stim(SortTraces(x1_stim),:))
xlabel('Time (s)')
ylabel('Units')
% colormap('Summer')
box off
xticks([0 1000 2000 3000 4000 5000 6000])
xticklabels({'0','1','2','3','4','5','6'})

%% ISI and ASI for single unit
[ISI_scale, ASI_scale] = SpAbScIndex(x1_scale,x2_scale);
[ISI_abs, ASI_abs] = SpAbScIndex(x1_abs(1:50,:),x2_abs(1:50,:));
[ISI_stim, ASI_stim] = SpAbScIndex(x1_stim,x2_stim);

edges_ISI = [0:0.025:1];
edges_ASI = [0:0.025:1];

subplot(5,3,10)
histogram(ISI_scale,edges_ISI,'Normalization', 'Probability')
box off
xlabel('Single-unit ISI')
ylabel('Probability')


subplot(5,3,11)
histogram(ISI_abs,edges_ISI,'Normalization', 'Probability')
box off
xlabel('Single-unit ISI')
ylabel('Probability')


subplot(5,3,12)
histogram(ISI_stim,edges_ISI,'Normalization', 'Probability')
box off
xlabel('Single-unit ISI')
ylabel('Probability')



subplot(5,3,13)
histogram(ASI_scale,edges_ASI,'Normalization', 'Probability')
box off
xlabel('Single-unit ASI')
ylabel('Probability')

subplot(5,3,14)
histogram(ASI_abs,edges_ASI,'Normalization', 'Probability')
box off
xlabel('Single-unit ASI')
ylabel('Probability')

subplot(5,3,15)
histogram(ASI_stim,edges_ASI,'Normalization', 'Probability')
box off
xlabel('Single-unit ASI')
ylabel('Probability')

%% ISI for population
[SSI_pop_scale] = SSI_pop(x1_scale,x2_scale);
[SSI_pop_abs] = SSI_pop(x1_abs(1:50,:),x2_abs(1:50,:));
[SSI_pop_stim] = SSI_pop(x1_stim,x2_stim);

figure(2)
X = categorical({'Scaling','Absolute','Stimulus-specific'});
X = reordercats(X,{'Scaling','Absolute','Stimulus-specific'});
bar(X,[SSI_pop_scale SSI_pop_abs SSI_pop_stim])
ylabel('Population SSI')


