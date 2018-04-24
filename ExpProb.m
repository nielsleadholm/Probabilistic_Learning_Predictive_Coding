function [x_estimate_all, mu_estimate, var_x_estimate, var_y_estimate, omega_estimate] ...
    = ExpProb(alpha_it, x_base, y_base)

%v3.0

%This function implements the explicit probabilistic model using analytical solutions to the x estimates.

alpha = alpha_it; %learning rate for variance and mu; note this is a vector
trials = length(x_base); %number of trials simulated

var_x_estimate = ones(1, trials); %the model's estimate of x-variance
var_y_estimate = ones(1, trials); %the model's estimate of y-variance
mu_estimate = ones(1, trials); %the model's estimate of the mu of x
omega_estimate = ones(1, trials); %model's estimate of omega (linear transformation in the distribution)
x_estimate_all = ones(1, trials); %stores x-estimate from each trial for later analysis and use

for j = 2:trials
    %set x and y values for improved human readability
    x = x_base(j);
    y = y_base(j);

    %x estimate is generated using analytical solution
    x_estimate_all(j) = (mu_estimate(j-1) * var_y_estimate(j-1) + omega_estimate (j-1) * y * var_x_estimate(j-1))...
        /(var_y_estimate(j-1) + (omega_estimate(j-1)^2) * var_x_estimate(j-1));

    %Based on the known (supervised) value of x, mu, omega and the
    %variances are updated
    mu_estimate(j) = mu_estimate(j-1) + alpha(1) * (x - mu_estimate(j-1))/var_x_estimate(j-1);
    var_y_estimate(j) = var_y_estimate(j-1) + alpha(2) * ((y - omega_estimate(j-1)*x)^2 - var_y_estimate(j-1));
    var_x_estimate(j) = var_x_estimate(j-1) + alpha(2) * ((x - mu_estimate(j-1))^2 - var_x_estimate(j-1));
    omega_estimate(j) = omega_estimate(j-1) + alpha(1) * x * (y - omega_estimate(j-1)*x)/var_y_estimate(j-1);

end
    