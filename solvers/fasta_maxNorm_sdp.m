%   Solve the max-norm least-squares problem
%           minimize_{X}  <S, X*X'>
%           subject to     MAXROW(X) <= mu
%   using the solver FASTA.  The operator MAXROW returns the norm of the
%   largest row appearing in X.  The inner product <.,.> returns the sum of
%   the coordinate-wise products of its arugments.
%
%  Inputs:
%    S   : A matrix of data, size NXN
%    X0  : Initial guess of solution, size NXK
%    mu  : Scalar regularization parameter
%    opts: Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.



function [ solution, outs ] = fasta_maxNorm_sdp( S, X0, mu, opts )

%%  Check that we have matrices
assert(isnumeric(S) & isnumeric(X0),'Inputs must be matrices.')
assert(size(S,1)==size(S,2),'Input S must be square.')
assert(size(S,1)==size(X0,1),'Invalid dimensions for inputs.')

%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end


%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  f(X) = <M,X*X'>
f    = @(X) sum(sum(S.*(X*X')));
grad = @(X) (S+S')*X;
A    = @(X) X;
At   = @(X) X;

g = @(Z) 0; 
prox = @(Z,t) projectOntoMaxNormBall(Z,mu);% proxg(Z,t):  Ensure all rows of X have norm less than mu

%% Call solver
[solution, outs] = fasta(A,At,f,grad,g,prox,X0,opts);

end


% Project Z onto a max norm ball with specific radius
function W = projectOntoMaxNormBall(Z, radius)
  norms = sqrt(sum(Z.*Z,2));  % compute norm of each row
  rescale = max(norms,radius);     % only renormalize columns that are too big
  rescale = rescale+(rescale==0);  % make sure we don't divide by zero
  
  %  Expand this to renormalize all columns Z
  numCols = size(Z,2);
  rescale = kron(rescale,ones(1,numCols));
 
  % Perform renormalization
  W = radius*(Z./rescale);
  
  %max(sum(W.*W,2))
  %norm(W-Z,'fro')
end
  
