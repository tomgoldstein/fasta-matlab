%   Solve the problem
%           min  mu||X||_nuc +.5||A(X)-b||^2 
%             X>=0
%   where X is a square symmetric matrix,||X||_nuc is the nuclear (trace) 
%   norm, A is a linear operator, and X>=0 denotes that X
%   must lie in the positive semidefinite cone.
%     The problem is solved using FASTA.  
%
%  Inputs:
%    A   : A measurement matrix that acts on X(:)
%    b   : Column vector of measurements
%    mu  : Scalar regularization parameter
%    opts: Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.



function [ solution, outs ] = fasta_phaselift( A,b,mu,X0,opts )

%% Check that inputs are valid
%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end

%  Make sure initial iterate is in column form
x0 = X0(:);
%  The dimension of the lifted matrix
n = sqrt(numel(X0));



%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  f(z) = .5 ||z - b||^2
f    = @(z) .5*norm(z-b,'fro')^2;
gradf = @(z) z-b;
% g(z) = mu||X||_nuc, plus characteristic function of the SDP cone
g  = @(x) mu*norm(eig( reshape(x,[n,n]) ),1);
% proxg(z,t) = argmin t*mu*nuc(x)+.5||x-z||^2, with x in SDP cone
proxg = @(z,t)  project_semiDefCone(z, mu*t);


%% Call solver
[solution, outs] = fasta(A,[],f,gradf,g,proxg,x0,opts);

end

function [ x ] = project_semiDefCone( x, delta )
[rows,cols] = size(x);
if rows==cols
   [V,D] = eig(x);
   D = max(real(D)-delta,0);
   x = V*D*V';
else 
    n = sqrt(numel(x));
    x = reshape(x,[n,n]);
    [V,D] = eig(x);
    D = max(real(D)-delta,0);
    x = V*D*V';
    x = x(:);
end
end



