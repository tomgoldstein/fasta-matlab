%  This script shows how to use the FASTA to solve
%                min  mu*MMV(X)+1/2|| AX-B  ||^2
%  Where A is an MxN matrix, B is an MxL vector of measurements, and X is
%  the NxL vector of unknowns.  The parameter 'mu' controls the strength of
%  the regularizer.

%% Define problem parameters
M = 20;   % number of measurements
N = 30;  % dimension of sparse signal
L = 10;
K = 7;    % signal sparsity   
mu = 1;  %  regularization parameter
sigma = 0.1;  %  The noise level in 'b'


fprintf('Testing MMV using N=%d, M=%d, L = %d\n', N, M, L);

%%  Create sparse signal
X = zeros(N,L);
perm = randperm(N);
X(perm(1:K),:) = randn(K,L);

%% Define Random Gaussian Matrix
%  Note: 'A' could be a function handle if we were using the FFT or DCT
A = randn(M,N);

%% Define observation vector
B = A*X;
B = B+sigma*randn(size(B)); % add noise 

%%  The initial iterate:  a guess at the solution
X0 = zeros(N,L);

%%  OPTIONAL:  give some extra instructions to FASTA using the 'opts' struct
opts = [];
%opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 

%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_mmv(A,A',B,mu,X0, opts);

% Turn on FISTA-type acceleration
opts.accelerate = true;
[sol, outs_accel] = fasta_mmv(A,A',B,mu,X0, opts);

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[sol, outs_fbs] = fasta_mmv(A,A',B,mu,X0, opts);


%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [300, 300, 400, 300]);
subplot(1,2,1);
imagesc(X);
title('True Solution');
subplot(1,2,2);
imagesc(sol);
title('Recovered Solution');

plotConvergenceCurves;