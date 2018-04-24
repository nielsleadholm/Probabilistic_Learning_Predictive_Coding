function [x_estimate, a_estimate, b_estimate] = ExpReg(alpha_it, x_base, y_base)

%v3.0

%This function implements the explicit regression model using analytical solutions to the x estimates.

alpha = alpha_it; %learning rates for variance and mu - note that this is a 2 value vector
trials = length(x_base); %number of trials simulated

%Set initial guesses and preallocate memory for variables
a_estimate = ones(1, trials); %the model's estimate of A (the correlation coefficient)
b_estimate = ones(1, trials); %the model's estimate of B (the intercept prior)
x_estimate = ones(1, trials); %the model's estimate of x
del_error = ones(1, trials); %the prediction error term

for j = 2:trials
    %Model gives x estimate based on current parameters of linear function
    x_estimate(j) = a_estimate(j-1) * y_base(j) + b_estimate(j-1);
    
    %Estimate is compared with true value
    del_error(j) = x_base(j) - x_estimate(j);
    
    %Parameters are updated based on prediction error
    a_estimate(j) = a_estimate(j-1) + alpha(1) * del_error(j) * y_base(j);
    b_estimate(j) = b_estimate(j-1) + alpha(2) * del_error(j);
   
end