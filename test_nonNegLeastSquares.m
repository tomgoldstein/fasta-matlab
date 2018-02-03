%  This script shows how to use FASTA to solve
%                min  1/2|| Ax-b  ||^2, subject to x>=0
%  Where A is an MxN matrix, b is an Mx1 vector of measurements, and x is
%  the Nx1 vector of unknowns.  The entries in x are constrained to be
%  non-negative.

%% Define problem parameters
M = 200;   % number of measurements
N = 1000;  % dimension of sparse signal
K = 10;    % signal sparsity   
sigma = 0.005;  %  The noise level in 'b'

fprintf('Testing non-negative least-squares using N=%d, M=%d\n',N,M);

%%  Create sparse signal
x = zeros(N,1);
perm = randperm(N);
x(perm(1:K)) = 1;

%% Define Random Gaussian Matrix
%  Note: 'A' could be a function handle if we were using the FFT or DCT
A = randn(M,N);
A = A/norm(A);   %  Normalize the matrix so that our value of 'mu' is fairly invariant to N

%% Define observation vector
b = A*x;
b = b+sigma*randn(size(b)); % add noise 

%%  The initial iterate:  a guess at the solution
x0 = zeros(N,1);

%%  OPTIONAL:  give some extra instructions to FASTA using the 'opts' struct
opts = [];
%opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 

%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_nonNegLeastSquares(A,A',b,x0, opts);

% Turn on FISTA-type acceleration
opts.accelerate = true;
[sol, outs_accel] = fasta_nonNegLeastSquares(A,A',b,x0, opts);

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[sol, outs_fbs] = fasta_nonNegLeastSquares(A,A',b,x0, opts);



%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [300, 300, 400, 300]);
stem(x,'red');
hold on;
stem(sol,'blue');
hold off;
legend('True','Recovered');
xlabel('Index');
ylabel('Signal Value');
title('True Sparse Signal');


plotConvergenceCurves;