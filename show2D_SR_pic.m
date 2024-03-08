% clc
load('E:\ZTE_SR\result.mat');
index = 6;
HR_target = squeeze(HR_Data(index,:,:));
LR_target = squeeze(LR_Data(index+1,:,:));

mask_high = squeeze(Origin(5,:,:));
HR_target(mask_high==1)=nan;
mask_low = squeeze(LR_Data(5,:,:));
LR_target(mask_low==1)=nan;

gr_tr = squeeze(Origin(index+1,:,:));
gr_tr(mask_high==1)=nan;
% K, phi,theta,power,delay,los 
% phi, theta, poweratio, power, delay, los

% poweratio(HR_target,gr_tr);
% phi(HR_target,gr_tr);
% theta(HR_target,gr_tr);
% power(HR_target,gr_tr);
% delay(HR_target,gr_tr);
los(HR_target,gr_tr);

function [] = phi(high,low)
high = high*180;low =low*180;
subplot 121
sub1 = pcolor(low);
shading interp
colormap jet
caxis([0,140]);
set(sub1, 'lineStyle','none');
xlabel("RMS \phi Spread of Low Resolution")
c = colorbar;
set(get(c,'Title'),'string','[ ° ]','FontSize',15);

subplot 122
sub2 = pcolor(high);
shading interp
colormap jet
caxis([0,140]);
set(sub2, 'LineStyle','none');
xlabel("RMS \phi Spread of High Resolution")
end

function [] = theta(high,low)
high = high*180;low =low*180;
subplot 121
sub1 = pcolor(low);
shading interp
colormap jet
caxis([0,23]);
set(sub1, 'lineStyle','none');
xlabel("RMS \theta Spread of Low Resolution")
c = colorbar;
set(get(c,'Title'),'string','[ ° ]','FontSize',15);

subplot 122
sub2 = pcolor(high);
shading interp
colormap jet
caxis([0,23]);
set(sub2, 'LineStyle','none');
xlabel("RMS \theta Spread of High Resolution")
end

function [] = poweratio(high,low)
high = high*100;low =low*100;
subplot 121
sub1 = pcolor(low);
shading interp
colormap jet
caxis([-40,0]);
set(sub1, 'lineStyle','none');
xlabel("Rp of Low Resolution")
c = colorbar;
set(get(c,'Title'),'string','[dB]','FontSize',15);

subplot 122
sub2 = pcolor(high);
shading interp
colormap jet
caxis([-40,0]);
set(sub2, 'LineStyle','none');
xlabel("Rp of High Resolution")
end

function [] = power(high,low)
high = high*100;
low = low*100;
subplot 121
sub1 = pcolor(low);
shading interp
colormap jet
caxis([-220,-60]);
set(sub1, 'LineStyle','none');
xlabel("Received Power of Low Resolution")
c = colorbar;
set(get(c,'Title'),'string','[dBm]','FontSize',15);

subplot 122
sub2 = pcolor(high);
shading interp
colormap jet
caxis([-220,-60]);
set(sub2, 'LineStyle','none');
xlabel("Received Power of High Resolution")
end

function [] = delay(high,low)
high = high*100;low =low*100;
subplot 121
sub1 = pcolor(low);
shading interp
colormap jet
caxis([0,110]);
set(sub1, 'lineStyle','none');
xlabel("RMS Delay Spread of Low Resolution")
c = colorbar;
set(get(c,'Title'),'string','[ns]','FontSize',15);

subplot 122
sub2 = pcolor(high);
shading interp
colormap jet
caxis([0,110]);
set(sub2, 'LineStyle','none');
xlabel("RMS Delay Spread of High Resolution")
end

function [] = los(high,low)
high = high*100;low =low*100;
subplot 121
sub1 = pcolor(low);
shading interp
colormap jet
caxis([-1,0]);
set(sub1, 'lineStyle','none');
xlabel("LOS/NLOS of Low Resolution")
c = colorbar;
set(get(c,'Title'),'string',{'LOS: -1';'NLOS:0'},'FontSize',15);

subplot 122
sub2 = pcolor(high);
shading interp
colormap jet
caxis([-1,0]);
set(sub2, 'LineStyle','none');
xlabel("LOS/NLOS of High Resolution")
end
