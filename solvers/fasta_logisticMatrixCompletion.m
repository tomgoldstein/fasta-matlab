%   Solve the problem
%           min  mu ||x||* +f(X,B)
%   where 
%         f(Z,B) = sum(log(1+exp(z)) - (b==1).*z,1)
%   is the logistic negative log-likelihood function, ||-||* donotes the
%   nuclear (or trace) norm, and 'B' is a matrix with values in the set 
%   {-1,1} that records either 'success' or 'failure' of a bernoulli trial.
%     The problem is solved using FASTA.  
%
%  Inputs:
%    B   : A matrix of measurements
%    mu  : Scalar regularization parameter
%    opts: Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.


function [ solution, outs ] = fasta_logisticMatrixCompletion( B,mu,opts )

%% Check that inputs are valid
%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end
% Check that 'b' is binary
if ~isreal(B) || ~isempty(find(abs(B)~=1,1))
  error('All entries in b must be +1 or -1');
end

%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
At = @(X) X; % Both A and At are simply identities
A  = @(X) X;
%  'f' is the log-likelihood 
f    = @(Z) sum(sum(log(1+exp(Z)) - (B==1).*Z));  
gradf = @(Z) (-B./(1+exp(B.*(Z))) );
% g(z) = mu*|z|
g = @(X)  mu*norm(svd(X),1);
% proxg(z,t) = argmin t*mu*nuc(x)+.5||x-z||^2
proxg = @(Z,t) prox_nuclearNorm(Z,mu*t);
% Initial guess
x0 = zeros(size(B));

%% Call solver
[solution, outs] = fasta(A,At,f,gradf,g,proxg,x0,opts);

end

function [ X ] = prox_nuclearNorm( X,t )
[U,S,V] = svd(X);
S = shrink(S,t);
X = U*S*V';
end

function [ x ] = shrink( x,tau )
 x = sign(x).*max(abs(x) - tau,0);
end


