%  Solve:
%                min_u  mu*TV(u)+1/2*||u-im||^2
%   where "im" is a noisy image or any dimension (1D, 2D, 3D, or higher), 
%   and "TV" represents the total-variation seminorm.  This code forms the 
%   dual problem, which is of the form
%         min_p || div(grad(p)) - im/mu ||^2
%           subject to ||p||_infinity<=1
%  Inputs:
%    im      : An N-D array with noisy image data
%    mu      : A scaling parameter that controls the strength of TV penalty
%    opts    : Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.


function [ denoised, outs ] = fasta_totalVariation( im, mu, opts )


%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end

%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).
%  f(z) = .5 ||z - (1.0/mu)*im||^2
lim = (1.0/mu)*im;
f    = @(z) .5*norm(z(:)-lim(:),'fro')^2;
fgrad = @(z) z-lim;
% g(z) = 0 for all feasible z, and infinity otherwise.  However, iterations
% always end with a feasbile z, so we need not consider the infinity case.
g = @(x) 0;
gprox = @(z,t) projectIsotropic(z);%min(max(z,-1),1);


%guess
x0 = zeros(size(grad(im)));

%% Call solver
[solution, outs] = fasta(@(x)div(x),@(x)grad(x),f,fgrad,g,gprox,x0,opts);

denoised =  im - div(solution)*mu;

end


function g = projectIsotropic( g )
dims = size(g);
%  Find the norm of the gradient at each point in space
normalizer = sqrt(sum(g.*g,ndims(g)));
%  Create a normalizer that will shrink the gradients to have magnitude at
%  most 1
normalizer = max(normalizer,1);
%  Make copies of this normalization so it can divide every entry in the
%  gradient
expander = ones(1,ndims(g));
expander(end) = dims(end);
normalizer = repmat(normalizer, expander);
%  Perform the normalization
g = g./normalizer;
end

%  The gradient of an N-dimensional array. The output array has size 
%  [size(x) ndims(x)].  Note that this output array has one more dimension
%  that the input array. This array contains all of the partial derivatives
%  (i.e., forward first-order differences) of x.  The partial derivatives
%  are indexed by the last dimension of the returned array.  For example, 
%  if x was 3 dimensional, the returned value of g has 4 dimensions.  The
%  x-derivative is stored in g(:,:,:,1), the y-derivative is g(:,:,:,2), 
%  and the z-derivative is g(:,:,:,3). 
%   Note:  This method uses circular boundary conditions for simplicity.
function [ g ] = grad( x )
    numdims = ndims(x);
    numEntries = numel(x);
    g = zeros(numEntries*numdims,1);
    for d = 1:numdims
        shift = zeros(1,numdims);
        shift(d) = 1;
        delta = circshift(x,shift)-x;
        g((d-1)*numEntries+1:d*numEntries) = delta(:);
    end
    g = reshape(g,[size(x), numdims]);
end


%  The divergence operator.  This method performs backward differences on
%  the input vector x.  It then sums these differences, and returns an
%  array with 1 dimension less than the input.  Note:  this operator is the
%  adjoint/transpose of "grad."
function [ out ] = div( x )
    s = size(x);
    dims = s(end);
    outdims = s(1:end-1);
    out = zeros(outdims);
    x = x(:);
    block = numel(out);
    for d = 1:dims
        slice = reshape(x(block*(d-1)+1:block*d), outdims);
        shift = zeros(1,dims);
        shift(d) = -1;
        out = out + circshift(slice,shift)-slice;
    end
end