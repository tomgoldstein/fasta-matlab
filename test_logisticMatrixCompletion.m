%  This script shows how to use FASTA to solve the 1-bit matrix completion
%  problem:
%                min  mu|x|+LL(Ax,b)
%  Where 'f' is the logistic negative log-liklihood function,
%  'A' is an MxN matrix, 'b' is an Mx1 vector of measurements, and 'x' is
%  the Nx1 vector of unknowns.  The parameter 'mu' controls the strength of
%  the regularizer.
%    Note the vector 'b' contains +1/-1 entries for success/failure of
%  Bernoulli trials.

%% Define problem parameters
M = 200;   % number of rows
N = 1000;   % number of cols
K = 10;      % rank of matrix
mu = 20;    %  regularization parameter

fprintf('Testing matrix completion with N=%d, M=%d\n',N,M);

%%  Create an MxN matrix of rank K
A = randn(M,N);
[U,S,V] = svd(A);
Shat = zeros(size(S));
Shat(1:K,1:K) = S(1:K,1:K);
A = U*Shat*V';
A = 10*A;

%% Define observations from random Bernoulli trials
sigmoid = @(X) exp(X)./(1+exp(X));
P = sigmoid(A); % Success probabilities
B = rand(M,N)<P; % convert probs to bernoulli trials
B = 2*B-1;   % convert to +-1

%%  The initial iterate:  a guess at the solution
X0 = zeros(M,N);

%%  OPTIONAL:  give some extra instructions to FASTA
opts = [];
opts.tol = 1e-5;     % Use strict tolerance
opts.verbose = true; % print convergence information on every iteration
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 
opts.recordObjective = true;

%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_logisticMatrixCompletion( B,mu,opts );

% Turn on FISTA-type acceleration
opts.accelerate = true;
[sol, outs_accel] = fasta_logisticMatrixCompletion( B,mu,opts );

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[sol, outs_fbs] = fasta_logisticMatrixCompletion( B,mu,opts );



%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure;
subplot(2,2,1);
imagesc(A);
title('True Matrix');

subplot(2,2,2);
imagesc(sol);
title('Recovered Matrix');

subplot(2,2,3);
plot(svd(sol));
title('Recovered Eigenvalues');

subplot(2,2,4);
semilogy(outs_adapt.residuals,'r');
hold on;
semilogy(outs_accel.residuals,'b');
semilogy(outs_fbs.residuals,'g');
xlabel('Iteration');
ylabel('residual norm');
legend('adaptive','accelerated','original');
title('residuals');
