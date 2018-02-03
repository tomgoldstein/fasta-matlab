%  This script shows how to use FASTA to solve the phase retrieval problem:
%           min  mu||X||_nuc +.5||A(X)-b||^2 
%             X>=0
%   where X is a square symmetric matrix,||X||_nuc is the nuclear (trace) 
%   norm, A is a linear operator, and X>=0 denotes that X must lie in the
%   positive semidefinite cone. The parameter 'mu' controls the strength of
%   the regularizer.
%     The code generates a complex vector x, and then its lifted
%   representation X=xx'.  It then creates a linear operator 'A' that acts
%   on the vectorized lifted representation X(:).  Measurements, 'b' are
%   created by computing |A*X(:)|.  Finally, the unknown vector is
%   recovered using FASTA.



%% Define problem parameters
m = 500;   % number of measurements
n = 100;   % dimension of signal to recover  
mu = .1;  %  The noise level in 'b'

% Number of elements in the lifted representation
N = n*n;

fprintf('Testing phaselift with n=%d, m=%d\n',n,m);

%% Create random complex vector, and lifted representation
x = randn(n,1)+randn(n,1)*1i;  %  random signal
X = x*x';  % lifted representation

%% Create measurement matrix that acts on the column vector X(:)
A = zeros(m,N);
for i=1:m
  a = randn(n,1)+randn(n,1)*1i;
  a = a/norm(a);
  a = a*a';
  A(i,:) = a(:);
end

%%  Create measurement vector
b = abs(A*X(:));  %  the measurements

%% Initial Guess
x0 = zeros(N,1);  % initial guess is zero

%%  OPTIONAL:  give some extra instructions to FASTA
opts = [];
opts.verbose = true;  
   % Note:  we don't set opts.recordObjective=true, because computing the
   % objective is expensive.  Evaluating g requires an SVD.
opts.maxIters = 1000;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 

   
%%  Call the solver 3 times
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_phaselift( A,b,mu,x0,opts);

% Turn on FISTA-type acceleration
opts.accelerate = true;
[sol, outs_accel] = fasta_phaselift( A,b,mu,x0,opts);

% Do plain old vanilla FBS
opts.accelerate = false;
opts.adaptive = false;
[sol, outs_fbs] = fasta_phaselift( A,b,mu,x0,opts);


%% Recover solution using method recommended by PhaseLift authors
%  The solution matrix may not be rank 1.  In this case, we use the
%  principle eigenvector and re-scale it to capture the correct ammount 
%  of energy.
[V,D] = eig(reshape(sol,[n,n])); %  get principle eigenvector
[val,ind] = max(diag(D));
recovered = V(:,ind)*sqrt(D(ind,ind));
      % figure out how muhc energy we're missing because the solution might
      % have other eigenvectors
lifted = recovered*recovered';
scale = norm(b)/norm(A*lifted(:));
       % scale the solution to capture the lost energy
lifted = lifted*scale;
recovered = recovered*scale;

%% Plot results
% This block allows plotting to be turned off by setting noPlots=true.
if exist('noPlots','var')  
    return;
end
figure('Position', [300, 300, 1000, 300]);
subplot(1,3,1);
scatter(abs(x),abs(recovered));
xlabel('True signal modulus');
ylabel('Recovered signal modulus');
title('True Vs. Recovered Signal');

subplot(1,3,2);
scatter(b,abs(A*lifted(:)));
xlabel('True signal modulus');
ylabel('Recovered signal modulus');
title('True Vs. Recovered Measurements');


subplot(1,3,3);
semilogy(outs_adapt.residuals,'r');
hold on;
semilogy(outs_accel.residuals,'b');
semilogy(outs_fbs.residuals,'g');
xlabel('Iteration');
ylabel('residual norm');
legend('adaptive','accelerated','original');
title('residuals');
