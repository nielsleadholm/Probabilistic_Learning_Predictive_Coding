function [x_estimate, mu_z_estimate, var_z_estimate, mu_estimate, var_estimate, omega_estimate] ...
    = Common(alpha_it, x_base, y_base)

%v3.0

%This function implements the common variable model using analytical solutions to the x and z estimates.

alpha = alpha_it; %learning rates for variance and mu
trials = length(x_base); %number of trials simulated

%Pre-allocate memory for model parameters 
x_estimate = ones(1, trials); %the model's best inferred x-estimate on each trial, prior to the supervised learning aspect
mu_estimate = ones(2, trials); %vector of model's mu estimates of x (1) and y (2)
mu_z_estimate = ones(1, trials); %the model's estimate of the mu of x
var_estimate = ones(2, trials); %vector of model's estimate of x (1) and y (2) variance
var_z_estimate = ones(1, trials); %model's estimate of z variance
omega_estimate = ones(2, trials); %model's estimate of omegas (linear transformations in the distribution)

for jj = 2:trials
    %u vector contains x-estimate (1) and y input (2)
    u(2) = y_base(jj); %Assign input y_value
    
    %steady state analytical solution for x-estimate (unsupervised)
    u(1) = mu_estimate(1, jj-1) + omega_estimate(1, jj-1) * ...
        (var_estimate(2, jj-1) * mu_z_estimate(jj-1) + var_z_estimate(jj-1) * (u(2) - mu_estimate(2, jj-1)) * omega_estimate(2, jj-1))...
        / (var_estimate(2, jj-1) + var_z_estimate(jj-1) * (omega_estimate(2, jj-1)^2));
    
    %Store model's x estimate for later use
    x_estimate(1, jj) = u(1);
    
    %Supervised aspect: x is set to true value
    u(1) = x_base(jj);
    
    %Analytical solution for z based on a known x value
    z_estimate = (var_estimate(2, jj-1) * var_z_estimate(jj-1) * omega_estimate(1, jj-1) * (u(1) - mu_estimate(1, jj-1)) ...
        + var_estimate(1, jj-1) * var_z_estimate(jj-1) * omega_estimate(2, jj-1) * (u(2) - mu_estimate(2, jj-1)) ...
        + var_estimate(1, jj-1) * var_estimate(2, jj-1) * mu_z_estimate(jj-1))...
        / (var_estimate(2, jj-1) * var_z_estimate(jj-1) * (omega_estimate(1, jj-1)^2)...
        + var_estimate(1, jj-1) * var_z_estimate(jj-1) * (omega_estimate(2, jj-1)^2)...
        + var_estimate(1, jj-1) * var_estimate(2, jj-1));

    %Update model estimates of mu, variance of x, y, and z, and omega,
    %using the analytical solution of z and the known x value
    mu_z_estimate(jj) = mu_z_estimate (jj-1) + alpha(1) * (z_estimate - mu_z_estimate(jj-1)) / var_z_estimate(jj-1);
    mu_estimate(:, jj) = mu_estimate(:, jj-1) + alpha(1) * ((u(:) - mu_estimate(:, jj-1))./var_estimate(:, jj-1));
    var_z_estimate(jj) = var_z_estimate(jj-1) + alpha(2) * ((mu_z_estimate(jj-1) - z_estimate)^2 - var_z_estimate(jj-1));
    var_estimate(:, jj) = var_estimate(:, jj-1) + alpha(2) * ((u(:) - mu_estimate(:, jj-1) - omega_estimate(:, jj-1) * z_estimate).^2 - var_estimate(:, jj-1)); %Note var_estimate includes x and y varainces
    omega_estimate(:, jj) = omega_estimate(:, jj-1) + alpha(1) * z_estimate * (u(:) - mu_estimate(:, jj-1) - omega_estimate(:, jj-1) * z_estimate) ./ var_estimate(:, jj-1);

end


