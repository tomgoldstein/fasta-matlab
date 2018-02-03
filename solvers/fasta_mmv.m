%   Solve the multiple measurement vector (MMV) problem
%           min  mu*MMV(x)+.5||Ax-B||^2
%   using the solver FASTA.  
%
%  Inputs:
%    A   : A matrix or function handle
%    At  : The adjoint/transpose of A
%    B   : A column vector of measurements
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



function [ solution, outs ] = fasta_mmv( A,At,B,mu,X0,opts )

%%  Check whether we have function handles or matrices
if ~isnumeric(A)
    assert(~isnumeric(At),'If A is a function handle, then At must be a handle as well.')
end
%  If we have matrices, create handles just to keep things uniform below
if isnumeric(A)
    At = @(X)A'*X;
    A = @(X) A*X;
end

%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end


%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  f(z) = .5 ||z - B||^2
f    = @(Z) .5*norm(Z-B,'fro')^2;
grad = @(Z) Z-B;
% g(z) = mu*MMV(Z)
g = @(X) mu*sum(sqrt(sum(X.*X,2)));
% proxg(z,t) = argmin t*mu*|x|+.5||x-z||^2
prox = @(X,t) shrink_rows(X,t*mu);

%% Call solver
[solution, outs] = fasta(A,At,f,grad,g,prox,X0,opts);

end


%%  The prox operator of MMV.  This function solves
%       min_X    tau*MMV(X)+0.5||X-Z||^2
function [ X ] = shrink_rows( Z, tau )
 [rows,cols] = size(Z);
 norms = sqrt(sum(Z.*Z,2));
 scale = max(norms - tau,0)./(norms+(norms==0));
 scale = kron(scale,ones(1,cols));
 X = Z.*scale;
end

