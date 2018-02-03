%  This script shows how to use FASTA to solve the problem:
%                min  mu*TV(u)+(1/2)||u-im||^2
%  Where "im" is a noisy image, and TV is the total
%  variation semi-norm.
%    Note:  The function "fasta_totalVariation" works when "im" has 
%    arbitrary dimension. It can denoise signals of dimension 1, 2, 3, or 
%    higher.  For demonstration, this script uses a 2D image.

%% Define problem parameters
mu = 0.1; % regularization parameter
N = 256;     % image dimension is NXN

image = phantom(N,N);

noisy = image+ randn(size(image))*.05;

fprintf('Testing democratic representations with N=%d, mu=%d\n',N,mu);

%%  OPTIONAL:  give some extra instructions to FASTA
opts = [];
opts.tol = 1e-3;  % Use custom stopping tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.maxIters = 1000;
opts.verbose = true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 


%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_totalVariation(noisy, mu, opts);

% Turn on FISTA-type acceleration
opts.accelerate = true;
[sol, outs_accel] = fasta_totalVariation(noisy, mu, opts);

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[sol, outs_fbs] = fasta_totalVariation(noisy, mu, opts);



%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [100, 300, 700, 300]);
subplot(1,2,1);
imagesc(noisy);
axis off;
title('noisy');

subplot(1,2,2);
imagesc(sol);
axis off;
title('denoised');

plotConvergenceCurves;