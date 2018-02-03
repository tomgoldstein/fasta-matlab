%  This script shows how to use FASTA to solve the problem:
%                min_w  0.5||w||+C*h(Dw,L)
%   where D is a matrix of observed vectors (one per row), L is a vector of
%   labels (one per row of D), h is the hinge loss function, and C is a 
%   regularization constant chosen by the user. 

%    Note:  The function "fasta_totalVariation" works when "im" has 
%    arbitrary dimension. It can denoise signals of dimension 1, 2, 3, or 
%    higher.  For demonstration, this script uses a 2D image.

%% Define problem parameters
M = 1000;  %  Number of observation vectors
N = 15;   %  Number of features per vector
C = .01;    %  Regularization parameter

%% Create two classes of data vectors that are linearly separable
class1 = 2*randn(M/2,N)-1;
class2 = 2*randn(M/2,N)+1;


%%  Create data matrix and labels
D = [class1;class2];
L = [ones(M/2,1);-ones(M/2,1)];
% Append a constant column to the data to allow for a bias
%D = [D 10*ones(M,1)];

fprintf('Testing SVM with N=%d, M=%d\n',N,M);

%%  OPTIONAL:  give some extra instructions to FASTA
opts = [];
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.maxIters = 5000;
%opts.stopRule = 'iterations';
opts.verbose = true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 


%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[w, outs_adapt] = fasta_svm( D, L, C, opts );

% Turn on FISTA-type acceleration
opts.accelerate = true;
[ ~ , outs_accel] = fasta_svm( D, L, C, opts );

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[ ~ , outs_fbs] = fasta_svm( D, L, C, opts );



%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
fprintf('Classification Accuracy: %f percent\n', sum(sign(D*w)==L)/M );
figure;
hist(D(1:M/2,:)*w,50);
h = findobj(gca,'Type','patch');
set(h,'FaceColor','r','EdgeColor','w','facealpha',0.75)
hold on;
hist(D(M/2+1:end,:)*w,50);
h1 = findobj(gca,'Type','patch');
set(h1,'facealpha',0.75);
title('hist(D*w)');


plotConvergenceCurves;