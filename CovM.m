function [x_estimate, mu, covar] ...
    = CovM(alpha_it, x_base, y_base)

%v3.0

%This function implements the covariance model using analytical solutions to the x estimates.

alpha = alpha_it; %learning rates for variance and mu
trials = length(x_base); %number of trials simulated

%Set initial guesses and preallocate memory for variables
u = ones(2, 1); %vector of the model's current estimate of x and y; note however that y in u(2, :) is 
%set and not altered by the model
covar = eye(2, 2); %covariance matrix of the model
mu = ones(2, trials); %the model's estimates of mu for both input variables in u
x_estimate = ones(1, trials); %the model's best inferred x-estimate on each trial, prior to the supervised learning aspect

for j = 2:trials
    %Assigns the input value of y to the u (model input) vector
    u(2) = y_base(j);
    
    %Creates inverse covariance matrix for simpler indexing later
    covar_inv = inv(covar);
    
    %Solves stable point value of x-estimate
    u(1) = (covar_inv(1,1) * mu(1, j-1) + covar_inv(1,2) * mu(2, j-1)...
        - covar_inv(1,2) * u(2)) / covar_inv(1,1);
    
    %store x_estimate for later analysis and use
    x_estimate(:, j) = u(1);
  
    %Supervised aspect - x node is set to true value
    u(1) = x_base(j);
    
    %Calculates stable point epsilon and inhib values based on the known
    %value of x
    Epsilon(:) = covar(:, :) \ (u(:) - mu(:, j-1));
    Inhib(:) = u(:) - mu(:, j-1);
    
    %Mu and the covariance matrix are then updated with these values
    mu(:, j) = mu(:, j-1) + alpha(1) * Epsilon(:);
    covar(:, :) = covar(:, :) + alpha(2) * (Epsilon(:) * Inhib(:)' - eye(2,2));

end

