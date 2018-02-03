%  This script shows how to use the FASTA to solve
%           minimize_{X}  <M, X'*X>
%           subject to     MAXROW(X) <= mu
%   using the solver FASTA.  The operator MAXROW returns the norm of the
%   largest row appearing in X.  The inner product <.,.> returns the sum of
%   the coordinate-wise products of its arugments.
%
%   The test problem is segmentation using the two-moons dataset and
%   a max-cut approximation.  This particular examples is described on page
%   7 of the article "Practical Large-Scale Optimization for Max-Norm 
%   Regularization."
%



%% Define problem parameters
N = 2000;   % Number of observations in the two moons dataset
D = 2;  % Dimensions of each observation vectors
sigma = .1; %  Distance parameter for similarity metric
noise = sqrt(0.02);
delta = 0.01;  % Balance parameter for the segmentation
K = 10;  %  Maximum allowed rank of factorization.  The max-cut problem uses a relaxation with factors of size NxK 

fprintf('Testing max norm SDP using N=%d, D=%d\n',N,D);

%% Build two-moons dataset for segmentation
theta = (0:(N-1))/N*2*pi;  %  Generate points on a circle
x = cos(theta);
y = sin(theta);
topMoon = y>0;
x(topMoon) = x(topMoon)-1;
y(topMoon) = y(topMoon)-.5;
    % Embed circle into D dimensions
data = zeros(N,D);
data(:,1) = x;
data(:,2) = y;
    %  Add noise
data = data + noise*randn(N,D);

% Get distance of each point to its nearest neighbor (excluding itself)
[~,nearest] = knnsearch(data,data,'K',2);
nearest = nearest(:,2);
dist = kron(nearest,ones(1,N));
dist = max(dist,dist');

%  Build similarity matrix
S = pdist2(data,data);
S = exp(-S.^2/sigma^2/2);


% Build the edge graph for the max-cut problem
Q = delta - S;



%%  CALL THE SOLVER

%  The initial iterate:  a guess at the solution
X0 = randn(N,K)/sqrt(K)/10;

%X0(:,1) = rand(N,1)>.5;

%%  OPTIONAL:  give some extra instructions to FASTA using the 'opts' struct
opts = [];
opts.tol = 1e-3;  % Use super strict tolerance
opts.maxIters = 1000;  
%opts.stopRule = 'iterations';  
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=1;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 

%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[ X1, outs_adapt ] = fasta_maxNorm_sdp( Q, X0, 1, opts );
% Turn on FISTA-type acceleration
opts.accelerate = true;
[ X2 , outs_accel ] = fasta_maxNorm_sdp( Q, X0, 1, opts );
% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[ X3 , outs_fbs ] = fasta_maxNorm_sdp( Q, X0, 1, opts );


%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [300, 300, 400, 300]);
% slice the solution with a random vector to create a graph cut
subplot(1,3,1);
slice = randn(K,1);
labels = sign(X1*slice);
%  Plot the points in each class
scatter(data(labels>=0,1),data(labels>=0,2),'b');
hold on;
scatter(data(labels<0,1),data(labels<0,2),'r');
hold off;
title('Adaptive');

subplot(1,3,2);
slice = randn(K,1);
labels = sign(X2*slice);
%  Plot the points in each class
scatter(data(labels>=0,1),data(labels>=0,2),'b');
hold on;
scatter(data(labels<0,1),data(labels<0,2),'r');
hold off;
title('Accelerated');

subplot(1,3,3);
slice = randn(K,1);
labels = sign(X3*slice);
%  Plot the points in each class
scatter(data(labels>=0,1),data(labels>=0,2),'b');
hold on;
scatter(data(labels<0,1),data(labels<0,2),'r');
hold off;
title('Original FBS');




plotConvergenceCurves;