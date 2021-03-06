function Model_recovery

%v3.0

%This code simulates noisy data generated by each of the four models, then
%attempts model recovery.

%Input parameters that model will learn through supervised technique:
%The following distribution follows the structure of the 'common variable'
%model outlined in section 4.5. _A and _B variables refer to the values
%these parameters take in the first and second half of the task.
params = struct; 
params.mu_z = 5; %mean of the base distribution
params.var_z_A = 1; %variance of base distribution
params.var_z_B = 1; %variance of the z variable
params.var_x_A = 0; %variance of the x variable
params.var_x_B = 0;
params.var_y = 1; %variance of the y variable
params.omega1A = 1; %linear tranformation in the data distribution
params.omega1B = 3;
params.omega2 = 1; %second linear tranformation in the data distribution
params.deca_operator = 0; %variable used to specify if discretised trial sets are desired. This should only
%be set to 1 if using the Model_performance function.

params.trial_max = 40; %number of trials simulated for each recovery
params.trial_start = params.trial_max; %determines which trial number the optimise function begins iterating at. This
%should always be the same as trial_max unless using the Model_performance
%function.

samples = 3; %the number of data samples to generate and attempt recovery on for each model
noise_levels = 6; %number of added noise levels incremented
recovery_matrix = zeros(4,4,noise_levels); %Confusion matrix used to store recoveries;
%There is a 4x4 matrix for each noise-level
%note columns correspond to the model-type of the data being presented,
%while rows correspond to the model-type selected by the recovery process;
%thus values along the diagonal represent correct recoveries. Each recovery
%adds a 1 to the corresponding row and column.

%Returns optimal learning rates for the models to use
[optimal_ExpProb, optimal_ExpReg, optimal_CovM, optimal_Common] = Optimise(params);

%Assigns returned values to the param. structure
params.alpha_ExpProb = optimal_ExpProb(end, :);
params.alpha_ExpReg = optimal_ExpReg(end, :);
params.alpha_CovM = optimal_CovM(end, :);
params.alpha_Common = optimal_Common(end, :);

%Parallel loop iterating through increasing levels of noise
parfor ll = 1:noise_levels

    %returns recovery matrix at specified noise level
    recovery_matrix(:, :, ll) = recovery_fun(samples, ll, params);

end

true_positives = zeros(4, noise_levels);

for ii = 1:4
    true_positives(ii, :) = (100/samples) * recovery_matrix(ii, ii, :);
end

plot(0:2:10, true_positives(1, :), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2)
hold on
plot(0:2:10, true_positives(2, :), 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2)
plot(0:2:10, true_positives(3, :), 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2)
plot(0:2:10, true_positives(4, :), 'Color', [0.3010, 0.7450, 0.9330], 'LineWidth', 2)
legend('Explicit Probabilistic Model', 'Explicit Regression Model', 'Covariance Model', 'Common Variable')
xlabel('Zero-Mean Gaussian Noise')
ylabel('Percentage Successful Recovery')
axis([0 10 0 100])
title('Successful Model Recovery vs. Added Noise')

%Performs model recovery 
function [temp] = recovery_fun(samples, ll, params)

temp = zeros(4, 4);  %temporary matrix that stores recovery results for later assignment in recovery_matrix

%Assign parameters for use
mu_z = params.mu_z; %mean of the base distribution
var_z_A = params.var_z_A; %variance of base distribution
var_z_B = params.var_z_B; %variance of the z variable
var_x_A = params.var_x_A; %variance of the x variable
var_x_B = params.var_x_B;
var_y = params.var_y; %variance of the y variable
omega1A = params.omega1A;  %linear tranformation in the data distribution
omega1B = params.omega1B;
omega2 = params.omega2; %second linear transformation
trial_max = params.trial_max; %number of trials simulated

%Assigns the optimal learning rates previously identified for each model
%given the underlying distribution to be learned
alpha_ExpProb = params.alpha_ExpProb;
alpha_ExpReg = params.alpha_ExpReg;
alpha_CovM = params.alpha_CovM;
alpha_Common = params.alpha_Common;

%Loops through the selected number of data samples; note that the underlying distribution
%of data is identical across sample sets, but the data is randomly
%generated for each sample
for jj = 1:samples

    noise = 2*(ll-1); %noise levels are incrementilly increased; determines amount of mean-0 Gaussian noise to add to the model estimates that are generated
    
    %Matrix of the x-estimates on which recovery will be attempted; rows
    %correspond to the explicit probabilistic, explicit regression,
    %covariance, and common variable models respectively
    x_estimate = zeros(4, trial_max);
    
    %Generates underlying data to be learned
    %This process is split such that dynamic data sets can be generated
    %where the parameters determining the distribution change mid-task
    z_base = [normrnd(mu_z, sqrt(var_z_A), trial_max/2, 1); normrnd(mu_z, sqrt(var_z_B), trial_max/2, 1)];
    x_base = ones(trial_max, 1);
    y_base = ones(trial_max, 1);
    for kk = 1:(trial_max/2)
        x_base(kk) = normrnd(omega1A*z_base(kk), sqrt(var_x_A));
        y_base(kk) = normrnd(omega2*z_base(kk), sqrt(var_y));
    end
    for kk = ((trial_max/2)+1):trial_max
        x_base(kk) = normrnd(omega1B*z_base(kk), sqrt(var_x_B));
        y_base(kk) = normrnd(omega2*z_base(kk), sqrt(var_y));
    end

    %Generates x-estimates from each model for later recovery
    x_estimate(1, :) = ExpProb(alpha_ExpProb, x_base, y_base);
    x_estimate(2, :) = ExpReg(alpha_ExpReg, x_base, y_base);
    x_estimate(3, :) = CovM(alpha_CovM, x_base, y_base);
    x_estimate(4, :) = Common(alpha_Common, x_base, y_base);

    %Adds mean-zero gaussian noise to x-estimate data
    x_estimate = x_estimate + normrnd(0, sqrt(noise), 4, trial_max);

    alpha_it = [0.001; 0.0001]; %the learning rate that fminsearch begins from
        
    %Loop iterating through the four different models
    for mm = 1:4
        %fminsearch identifies the alpha value that minimises the cost
        %function, and returns the cost value at this point
        [~, min_cost_ExpProb] = fminsearch(@(alpha_it)cost_fun_ExpProb(alpha_it, x_base, y_base, x_estimate(mm, :)), alpha_it);
        [~, min_cost_ExpReg] = fminsearch(@(alpha_it)cost_fun_ExpReg(alpha_it, x_base, y_base, x_estimate(mm, :)), alpha_it);
        [~, min_cost_CovM] = fminsearch(@(alpha_it)cost_fun_CovM(alpha_it, x_base, y_base, x_estimate(mm, :)), alpha_it);
        [~, min_cost_Common] = fminsearch(@(alpha_it)cost_fun_Common(alpha_it, x_base, y_base, x_estimate(mm, :)), alpha_it);

        %Compares the cost results of the different models, assigning the
        %recovery to the model with the lowest value
        if  min_cost_ExpProb < min_cost_ExpReg && min_cost_ExpProb < min_cost_CovM && min_cost_ExpProb < min_cost_Common
            temp(1, mm) = temp(1, mm) + 1;
        elseif min_cost_ExpReg < min_cost_CovM && min_cost_ExpReg < min_cost_Common
            temp(2, mm) = temp(2, mm) + 1;
        elseif min_cost_CovM < min_cost_Common
            temp(3, mm) = temp(3, mm) + 1;
        else
            temp(4, mm) = temp(4, mm) + 1;
        end

    end
end

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_ExpProb(alpha_it, x_base, y_base, x_unknown)

x_sim = ExpProb(alpha_it, x_base, y_base);

%simulated data is then compared to the estimates of the unknown model
cost = sum(((x_sim - x_unknown).^2));

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_ExpReg(alpha_it, x_base, y_base, x_unknown)

x_sim = ExpReg(alpha_it, x_base, y_base);

%simulated data is then compared to the estimates of the unknown model
cost = sum(((x_sim - x_unknown).^2));

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_CovM(alpha_it, x_base, y_base, x_unknown)

x_sim = CovM(alpha_it, x_base, y_base);

%simulated data is then compared to the estimates of the unknown model
cost = sum(((x_sim - x_unknown).^2));

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_Common(alpha_it, x_base, y_base, x_unknown)

x_sim = Common(alpha_it, x_base, y_base);

%simulated data is then compared to the estimates of the unknown model
cost = sum(((x_sim - x_unknown).^2)); 
