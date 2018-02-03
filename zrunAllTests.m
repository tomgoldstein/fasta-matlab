% This will run all test scripts for the solvers included in the 
% FASTA distribution.  Each test is run 100 times on random problem 
% instances.  The runtime and iteration count is recorded for each variant
% of FBS implemented in FASTA.  The results are used to generate the 
% performance table in the review article "A field guide to forward-
% backward splitting with a FASTA implementation."
%  Note:  the folder "sovlers" must be in your PATH for this to work.

%% Set the number of times to run each test.  The performance of each algorithm is averaged over all trials
numberOfTrials = 100;

%%  Allocate space to store the iteration count and runtimes of each algorithm
lasso100_runtime = zeros(1,3);
lasso100_iters = zeros(1,3);
lasso500_runtime = zeros(1,3);
lasso500_iters = zeros(1,3);
bpdn100_runtime = zeros(1,3);
bpdn100_iters = zeros(1,3);
bpdn500_runtime = zeros(1,3);
bpdn500_iters = zeros(1,3);
logreg_runtime = zeros(1,3);
logreg_iters = zeros(1,3);
mmv_runtime = zeros(1,3);
mmv_iters = zeros(1,3);
democratic_runtime = zeros(1,3);
democratic_iters = zeros(1,3);
matcomp_runtime = zeros(1,3);
matcomp_iters = zeros(1,3);
tv_runtime = zeros(1,3);
tv_iters = zeros(1,3);
svm_runtime = zeros(1,3);
svm_iters = zeros(1,3);
phaselift_runtime = zeros(1,3);
phaselift_iters = zeros(1,3);
nmf_runtime = zeros(1,3);
nmf_iters = zeros(1,3);
maxnorm_runtime = zeros(1,3);
maxnorm_iters = zeros(1,3);


%%  This function prints the current average iterations count and runtime for each method
printResults = @(name, iters, runtime)  fprintf('%s &\t%5.0f (%0.3f) & \t%5.0f (%0.3f) & \t%5.0f (%0.3f) \\\\\n',name,...
    mean(iters(:,1)),mean(runtime(:,1)),...
    mean(iters(:,2)),mean(runtime(:,2)),...
    mean(iters(:,3)),mean(runtime(:,3)));

%% Turn off plotting of convergence curves
noPlots = true;  

%% Seed random number generator to make results reproducable
rng(0);  %  NOTE:  rng() does not exist in old versions of matlab.  Some users may need to comment this line out.

%%  Test that FASTA is in the PATH
assert(exist('fasta') && exist('fasta_lasso') && exist('fasta_totalVariation'),...
        'Error:  add the "solvers" folder to your PATH before running this script');

%%  Run each experiment 100 times
for trialNum = 1:numberOfTrials
    
fprintf('Begin Iteration %d\n',trialNum);
 
%% Phaselift - phase recovery from Fourier data
test_phaselift;
phaselift_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
phaselift_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%% Logistic Matrix Completion
test_logisticMatrixCompletion;
matcomp_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
matcomp_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%% Total Variation image denoising
test_totalVariation;
tv_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
tv_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%%  Sparse least-squares:  100X1000 Matrix
M_automated = 100;  % test with 100X1000 matrix
test_sparseLeastSquares;
bpdn100_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
bpdn100_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%%  Sparse least-squares:  500X1000 Matrix
M_automated = 500;   % test with 500X1000 matrix
test_sparseLeastSquares;
bpdn500_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
bpdn500_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%%  LASSO: 100X1000 Matrix
M_automated = 100;   % test with 100X1000 matrix
test_lasso;
lasso100_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
lasso100_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%%  LASSO: 500X1000 Matrix
M_automated = 500;   % test with 500X1000 matrix
test_lasso;
lasso500_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
lasso500_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%% Democratic representations
test_democratic;
democratic_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
democratic_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%% Sparse logistic regression
test_sparseLogistic;
logreg_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
logreg_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%% MMV regression
test_mmv;
mmv_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
mmv_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];

%% SVM regression
test_svm;
svm_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
svm_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];


%% nmf regression
test_nonNegFactorization;
nmf_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
nmf_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];


%% Max-Norm regression
test_maxNorm_sdp;
maxnorm_runtime(trialNum,:) = [outs_fbs.solveTime, outs_accel.solveTime, outs_adapt.solveTime];
maxnorm_iters(trialNum,:)   = [outs_fbs.iterationCount, outs_accel.iterationCount, outs_adapt.iterationCount];


%% Print average performance for all completed trials
fprintf('\nResults: %d/%d trials completed\n', trialNum, numberOfTrials);
fprintf('-------------------------------------------------------------------\n');
fprintf('                      FBS         ACCELERATED       ADAPTIVE\n');
fprintf('-------------------------------------------------------------------\n');
printResults('Lasso 100', lasso100_iters, lasso100_runtime);
printResults('Lasso 500', lasso500_iters, lasso500_runtime);
printResults('BPDN 100', bpdn100_iters, bpdn100_runtime);
printResults('BPDN 500', bpdn500_iters, bpdn500_runtime);
printResults('Logistic', logreg_iters, logreg_runtime);
printResults('MMV', mmv_iters, mmv_runtime);
printResults('Democratic', democratic_iters, democratic_runtime);
printResults('Mat Comp', matcomp_iters, matcomp_runtime);
printResults('TV Denoising', tv_iters, tv_runtime);
printResults('SVM', svm_iters, svm_runtime);
printResults('PhaseLift', phaselift_iters, phaselift_runtime);
printResults('NMF', nmf_iters, nmf_runtime);
printResults('Max-Norm', maxnorm_iters, maxnorm_runtime);
fprintf('-------------------------------------------------------------------\n\n');

end


