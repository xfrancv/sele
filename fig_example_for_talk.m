%%
% It generates EPS figures from paper to figs/ folder.

outFolder = 'figs/';


R1 = load( 'results/msvmlin/ijcnn1/results.mat' );
R2 = load( 'results/msvmlin/ijcnn1/conf_hinge1_quad_zmuv1_th5/results.mat');

textYoffset = 0.1;
fontSize = 15;
lineWidth = 3;
textX = 5;
P = [0.1744 0.7722 0.5804 0.0619];

%% figure 1: SVM + dist to hyperplane
hf1 = figure;
hold on;
    
nTst = size( R1.tstRiskCurve,1);
h    = plot( 100*[1:nTst]/nTst, 100*mean( R1.tstRiskCurve, 2), 'k', 'linewidth', lineWidth);
hold on;
grid on;
xlabel('coverage [%]');
ylabel('selective risk [%]');
hl=legend( h, 'SVM + distance to hyperplane' ,'Location', 'NorthWest' );
%hl.Position = P; 
a = axis;
%axis([0.5 a(2) minR50 maxR100]);
h=gca;
h.FontSize=16;    
print( hf1, '-depsc', 'figs/talk_fig1.eps');


%% figure 2: SVM + dist to hyperplane + wp1
hf1 = figure;
hold on;
    
nTst = size( R1.tstRiskCurve,1);
h    = plot( 100*[1:nTst]/nTst, 100*mean( R1.tstRiskCurve, 2), 'k', 'linewidth', lineWidth);
hold on;
grid on;
risk = 100*mean( R1.tstRiskCurve(end,:), 2);
plot(  100, risk, 'ok', 'markersize', 12,'markerfacecolor','k');
plot( [0 100], risk*[ 1 1],'--k', 'linewidth',lineWidth);
plot( [100 100], [risk 0],'--k', 'linewidth',lineWidth);
xlabel('coverage [%]');
ylabel('selective risk [%]');
hl=legend( h, 'SVM + distance to hyperplane' ,'Location', 'NorthWest' );
%hl.Position = P;
a = axis;
%axis([0.5 a(2) minR50 maxR100]);
h=gca;
h.FontSize=fontSize;    
print( hf1, '-depsc', 'figs/talk_fig1a.eps');

%% figure 3: SVM + dist to hyperplane + wp2
hf1 = figure;
hold on;
    
nTst = size( R1.tstRiskCurve,1);
h    = plot( 100*[1:nTst]/nTst, 100*mean( R1.tstRiskCurve, 2), 'k', 'linewidth', lineWidth);
hold on;
grid on;
idx = round(0.8* size(R1.tstRiskCurve,1));
risk = 100*mean( R1.tstRiskCurve(idx,:), 2);
plot(  80, risk, 'ok', 'markersize', 12,'markerfacecolor','k');
plot( [0 80], risk*[ 1 1],'--k', 'linewidth',lineWidth);
plot( [80 80], [risk 0],'--k', 'linewidth',lineWidth);
text( textX, risk+textYoffset, sprintf('R_S = %.1f%%', risk), 'VerticalAlignment', 'bottom','FontSize', fontSize);
xlabel('coverage [%]');
ylabel('selective risk [%]');
hl=legend( h, 'SVM + distance to hyperplane' ,'Location', 'NorthWest' );
%hl.Position = P;
a = axis;
%axis([0.5 a(2) minR50 maxR100]);
h=gca;
h.FontSize=15;    
print( hf1, '-depsc', 'figs/talk_fig1b.eps');

%% figure 4: SVM + dist to hyperplane
hf2 =figure;
hold on;
    
nTst = size( R1.tstRiskCurve,1);
h1   = plot( 100*[1:nTst]/nTst, 100*mean( R1.tstRiskCurve, 2), 'k', 'linewidth', lineWidth);
hold on;
grid on;
nTst = size( R2.tstRiskCurve,1);
h2   = plot( 100*[1:nTst]/nTst, 100*mean( R2.tstRiskCurve, 2), 'r', 'linewidth', lineWidth);

xlabel('coverage [%]');
ylabel('selective risk [%]');
hl=legend( [h1 h2], 'SVM + distance to hyperplane', 'SVM + learned selection function', 'Location', 'NorthWest' );
%hl.Position = P;
a = axis;
h = gca;
h.FontSize=fontSize;    
print( hf2, '-depsc', 'figs/talk_fig2.eps');

%% figure 5: SVM + dist to hyperplane
hf2 =figure;
hold on;
    
nTst = size( R1.tstRiskCurve,1);
h1   = plot( 100*[1:nTst]/nTst, 100*mean( R1.tstRiskCurve, 2), 'k', 'linewidth', lineWidth);
hold on;
grid on;
idx = round(0.8* size(R1.tstRiskCurve,1));
risk = 100*mean( R1.tstRiskCurve(idx,:), 2);
plot(  80, risk, 'ok', 'markersize', 12,'markerfacecolor','k');
plot( [0 80], risk*[ 1 1],'--k', 'linewidth',lineWidth);
plot( [80 80], [risk 0],'--k', 'linewidth',lineWidth);
text( textX, risk+textYoffset, sprintf('R_S = %.1f%%', risk), 'VerticalAlignment', 'bottom','FontSize', fontSize);

nTst = size( R2.tstRiskCurve,1);
h2   = plot( 100*[1:nTst]/nTst, 100*mean( R2.tstRiskCurve, 2), 'r', 'linewidth', lineWidth);
idx = round(0.8* size(R1.tstRiskCurve,1));
risk = 100*mean( R2.tstRiskCurve(idx,:), 2);
plot(  80, risk, 'or', 'markersize', 12,'markerfacecolor','r');
plot( [0 80], risk*[ 1 1],'--r', 'linewidth',lineWidth);
plot( [80 80], [risk 0],'--r', 'linewidth',lineWidth);
text( textX, risk+textYoffset, sprintf('R_S = %.1f%%', risk), 'VerticalAlignment', 'bottom','FontSize', fontSize,'color','r');

xlabel('coverage [%]');
ylabel('selective risk [%]');
hl=legend( [h1 h2], 'SVM + distance to hyperplane', 'SVM + learned selection function', 'Location', 'NorthWest' );
%hl.Position = P;
a = axis;
h = gca;
h.FontSize=fontSize;    
print( hf2, '-depsc', 'figs/talk_fig2a.eps');
