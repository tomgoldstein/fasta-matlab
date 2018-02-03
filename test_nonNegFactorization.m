%  This script shows how to use FASTA to solve
%           minimize_{X,Y}  mu|X|+.5||S-XY'||^2
%           subject to  norm(Y,'inf')<=1
%  Where S is an MxN matrix of data, X is an MXK matrix, and Y is a NXK
%  matrix.  The parameter 'mu' controls the strength of
%  the L1 regularizer.

%% Define problem parameters
M = 800;   % rows of data matrix
N = 200;  % cols of data matrix
K = 10;    % rank of factorization
mu = 1;

fprintf('Testing non-negative factorization using N=%d, M=%d, K=%d\n',N,M,K);

%%  Create non-negative factors
X = rand(M,K);
Y = rand(N,K);
% Make X 75% sparse
X = X.*(rand(M,K)>.75);
% Create observation/data matrix
S = X*Y';
S = S+randn(size(S))*0.1;

%%  The initial iterate:  a guess at the solution
X0 = zeros(M,K);
Y0 = rand(N,K);

%%  OPTIONAL:  give some extra instructions to FASTA using the 'opts' struct
opts = [];
%opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 


%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[ Xsol,Ysol, outs_adapt ] = fasta_nonNegativeFactorization( S, X0, Y0, mu, opts );

% Turn on FISTA-type acceleration
opts.accelerate = true;
[ Xsol,Ysol, outs_accel ] = fasta_nonNegativeFactorization( S, X0, Y0, mu, opts );

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[ Xsol,Ysol, outs_fbs ] = fasta_nonNegativeFactorization( S, X0, Y0, mu, opts );



%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [300, 300, 400, 300]);
subplot(2,2,1)
imagesc(X); title('Xtrue');
subplot(2,2,2)
imagesc(Y); title('Ytrue');
subplot(2,2,3)
imagesc(Xsol); title('Xrecovered');
subplot(2,2,4)
imagesc(Ysol); title('Yrecovered');

plotConvergenceCurves;