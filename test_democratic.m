%  This script uses FASTA to solve the democratic representation
%  problem:
%                min  mu ||x||_inf+1/2|| Ax-b  ||^2
%  Where A is an MxN matrix, b is an Mx1 vector of measurements, x is
%  the Nx1 vector of unknowns, and ||-||_inf is the infinity norm.
%  The parameter 'mu' controls the strength of the regularizer.

%% Define problem parameters
M = 500;   % number of original signal
N = 1000;  % dimension of democratic representation
mu = 300;  % regularization parameter

fprintf('Testing democratic representations with N=%d, M=%d\n',N,M);

%% Create a frame
% choose a random set of DCT modes to sample
ind = randperm(N-1)+1;
samps = ind(1:M-1);
samps = [1 sort(samps)]; % note: always sample the DC mode
%% Create subsampled DCT
R = zeros(N,1);
R(samps) = 1;
A = @(x) R.*dct(x);
At = @(x) idct(R.*x);

%% Create random signal
%  Note:  the M unknown measurements correspond to M rows of the DCT.  We
%  place the measurements at M random locations.
b = zeros(N,1);
b(samps) = randn(M,1);

%% Initial Guess
x0 = zeros(N,1);

%%  OPTIONAL:  give some extra instructions to FASTA
opts = [];
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose = 1;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 

%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_democratic(A,At,b,mu,x0, opts);

% Turn on FISTA-type acceleration
opts.accelerate = true;
[sol, outs_accel] = fasta_democratic(A,At,b,mu,x0, opts);

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[sol, outs_fbs] = fasta_democratic(A,At,b,mu,x0, opts);




%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [300, 300, 500, 300]);
plot(b,'ro');
hold on;
plot(sol,'b');
hold off;
legend('Original','Democracy');
xlabel('Index');
ylabel('Signal Value');
title('True Sparse Signal');


plotConvergenceCurves;