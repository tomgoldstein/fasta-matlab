%   Solve the L1 penalized logistic regression problem
%           min  mu|x|+f(Ax,b)
%   where 
%         f(z,b) = sum(log(1+exp(z)) - (b==1).*z,1)
%   is the logistic negative log-likelihood function, and 'b' is a vector 
%   with values in the set {-1,1} that records either 'success' or 
%   'failure' of a bernoulli trial.
%     The problem is solved using FASTA.  
%
%  Inputs:
%    A   : A matrix or function handle
%    At  : The adjoint/transpose of A
%    b   : A column vector of measurements
%    mu  : Scalar regularization parameter
%    x0  : Initial guess of solution, often just a vector of zeros
%    opts: Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.



function [ solution, outs ] = fasta_sparseLogistic( A,At,b,mu,x0,opts )

%% Check that inputs are valid
%  Check whether we have function handles or matrices
if ~isnumeric(A)
    assert(~isnumeric(At),'If A is a function handle, then At must be a handle as well.')
end
%  If we have matrices, create handles just to keep things uniform below
if isnumeric(A)
    At = @(x)A'*x;
    A = @(x) A*x;
end
%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end
% Check that 'b' is binary
if ~isreal(b) || ~isempty(find(abs(b)~=1,1))
  error('All entries in b must be +1 or -1');
end

%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  'f' is the log-likelihood 
f    = @(z) sum(log(1+exp(z)) - (b==1).*z,1);  
gradf = @(z) (-b./(1+exp(b.*(z))) );
% g(z) = mu*|z|
g = @(x) norm(x,1)*mu;
% proxg(z,t) = argmin t*mu*|x|+.5||x-z||^2
proxg = @(x,t) shrink(x,t*mu);

%% Call solver
[solution, outs] = fasta(A,At,f,gradf,g,proxg,x0,opts);

end


%%  The vector shrink operator
function [ x ] = shrink( x,tau )
 x = sign(x).*max(abs(x) - tau,0);
end

