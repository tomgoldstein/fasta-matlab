%  Solve the support vector machine fitting problem
%                min_w  0.5||w||+C*h(Dw,L)
%   where D is a matrix of observed vectors (one per row), L is a vector of
%   labels (one per row of D), h is the hinge loss function, and C is a 
%   regularization constant chosen by the user.  The solution is obtained 
%   by solving the dual problem
%         min_x || DL.x ||^2 - sum(x)
%           subject to 0<=x<=C
%  Inputs:
%    D      : An MxN array with M observation vectors
%    L      : An Mx1 vector of labels.  All entries are +1 or -1
%    C      : A scalar smoothing/regularization constant 
%    opts   : Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.


function [ w, outs, solution ] = fasta_svm( D, L, C, opts )


%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end

[m,n] = size(D);

%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  f(z) = .5 ||y||^2 - sum(x)
A = @(z) [D'*(L.*z); z];
At = @(z) L.*(D*z(1:n)) + z(n+1:end);
f    = @(z) .5*norm(z(1:n),'fro')^2-sum(z(n+1:end));
fgrad = @(z) [z(1:n);-ones(m,1)];
% g(z) = 0 for all feasible z, and infinity otherwise.  However, iterations
% always end with a feasbile z, so we need not consider the infinity case.
g = @(x) 0;
gprox = @(z,t) min(max(z,0),C);

%guess
x0 = zeros(m,1);

%% Call solver
[solution, outs] = fasta(A,At,f,fgrad,g,gprox,x0,opts);

w  =  D'*(L.*solution);

end
