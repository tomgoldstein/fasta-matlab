titleSize = 14;
labelSize = 14;


figure('Position', [500, 400, 700, 300]);
subplot(1,2,1);
plot(outs_fbs.objective,'g');
hold on;
plot(outs_accel.objective,'b');
plot(outs_adapt.objective,'r');
hold off;
objXLabel = xlabel('Iteration');
objYLabel = ylabel('Objective Value');
objLegend = legend('original FBS', 'accelerated', 'adaptive');
objTitle = title('Objective Function');

%set([objXLabel, objYLabel, objLegend, objTitle],'Interpreter','latex')
%set([objXLabel, objYLabel, objLegend],'Fontsize',13)
%set([objTitle],'Fontsize',14)

subplot(1,2,2);
semilogy(outs_fbs.residuals,'g');
hold on;
semilogy(outs_accel.residuals,'b');
semilogy(outs_adapt.residuals,'r');
hold off;
rXLabel = xlabel('Iteration');
rYLabel = ylabel('residual norm');
rLegend = legend('original FBS', 'accelerated', 'adaptive');
rTitle = title('Residuals');

%set([rXLabel, rYLabel, rLegend, rTitle],'Interpreter','latex')
%set([rXLabel, rYLabel, rLegend],'Fontsize',13)
%set([rTitle],'Fontsize',14)