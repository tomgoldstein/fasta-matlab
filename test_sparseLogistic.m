%  This script shows how to use the FASTA to solve
%                min  mu|x|+LL(Ax,b)
%  Where 'f' is the logistic negative log-liklihood function,
%  'A' is an MxN matrix, 'b' is an Mx1 vector of measurements, and 'x' is
%  the Nx1 vector of unknowns.  The parameter 'mu' controls the strength of
%  the regularizer.
%    Note the vector 'b' contains +1/-1 entries for success/failure of
%  bernoulli trials.

%% Define problem parameters
M = 1000;   % number of measurements
N = 2000;   % dimension of sparse signal
K = 5;      % signal sparsity   
mu = 40;    %  regularization parameter

fprintf('Testing logistic regression with N=%d, M=%d\n',N,M);

%%  Create sparse signal
x = zeros(N,1);
perm = randperm(N);
x(perm(1:K)) = 1;

%% Define Random Gaussian Matrix
%  Note: 'A' could be a function handle if we were using the FFT or DCT
A = randn(M,N);

%% Define observation vector
probSuccess = exp(A*x)./(1+exp(A*x)); % use logistic model to find probability of success
b = rand(M,1)<probSuccess;   %  create 0/1 vector of random trials
b = 2*b-1;                   %  convert to +/-1 vector

%%  The initial iterate:  a guess at the solution
x0 = zeros(N,1);

%%  OPTIONAL:  give some extra instructions to FASTA
opts = [];
opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose = 1;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 

%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_sparseLogistic(A,[],b,mu,x0, opts);

% Turn on FISTA-type acceleration
opts.accelerate = true;
[sol, outs_accel] = fasta_sparseLogistic(A,[],b,mu,x0, opts);

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[sol, outs_fbs] = fasta_sparseLogistic(A,[],b,mu,x0, opts);


%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [300, 300, 500, 300]);
stem(x,'red');
hold on;
stem(sol,'blue');
hold off;
legend('True','Recovered');
xlabel('Index');
ylabel('Signal Value');
title('True Sparse Signal');

plotConvergenceCurves;