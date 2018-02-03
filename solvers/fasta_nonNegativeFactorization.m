%   Solve the L1-penalized non-negative least-squares problem
%           minimize_{X,Y}  mu|X|+.5||S-XY'||^2
%           subject to  norm(Y,'inf')<=1, X>=0, Y>=0
%   using the solver FASTA.  Note that choosing mu=0 yields standard
%   non-negative least squares without an L1 penality.
%
%  Inputs:
%    S   : A matrix of data, size MXN
%    X0  : Initial guess of solution, size MXK
%    Y0  : Initial guess of solution, size NXK
%    mu  : Scalar regularization parameter
%    opts: Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.



function [ Xsol,Ysol, outs ] = fasta_nonNegativeFactorization( S, X0, Y0, mu, opts )

%%  Check that we have matrices
assert(isnumeric(S) & isnumeric(X0) & isnumeric(Y0),'Inputs must be matrices.')

%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end

% Make unknowns into a single large matrix so that FASTA can handle them
Z0 = [X0;Y0];
Xrows = 1:size(X0,1);
Yrows = size(X0,1)+1:size(X0,1)+size(Y0,1);

%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  f(Z) = .5 ||XY' - S||^2
f    = @(Z) .5*norm(Z(Xrows,:)*Z(Yrows,:)' - S,'fro')^2;
A = @(x) x;
At = @(x) x;
grad = @(Z) gradZ(Z,S,Xrows,Yrows);


% g(z) = mu*|z|
g = @(Z) 0;%matrix1norm(Z(Xrows,:))*mu;
% proxg(Z,t):  Perform shrink on X, and ensure Z is positive.  Ensure Y
% lies in [0,1]
prox = @(Z,t) [max(Z(Xrows,:)-t*mu,0) ; min(max(Z(Yrows,:),0),1)];

%% Call solver
[solution, outs] = fasta(A,At,f,grad,g,prox,Z0,opts);

Xsol = solution(Xrows,:);
Ysol = solution(Yrows,:);

end

function grad = gradZ(Z,S,Xrows,Yrows)
X = Z(Xrows,:);
Y = Z(Yrows,:);
diff = X*Y'-S;
dX = diff*Y;
dY = diff'*X;
grad = [dX;dY];
end 

function norm1 = matrix1norm(Z)
norm1 = sum(abs(Z(:)));
end 



