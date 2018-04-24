function [optimal_ExpProb, optimal_ExpReg, optimal_CovM, optimal_Common,...
        ExpProb_RMSD, ExpReg_RMSD, CovM_RMSD, Common_RMSD]... 
       = Optimise(params)

%v3.0

%Optimise is used by other functions to identify the learning rates for
%each model that minimise a cost function determined by the accuracy of their
%predictions

mu_z = params.mu_z; %mean of the base distribution
var_z_A = params.var_z_A; %variance of base distribution
var_z_B = params.var_z_B; %variance of the z variable
var_x_A = params.var_x_A; %variance of the x variable
var_x_B = params.var_x_B;
var_y = params.var_y; %variance of the y variable
omega1A = params.omega1A; %linear tranformation in the data distribution
omega1B = params.omega1B;
omega2 = params.omega2; %second linear tranformation in the data distribution

trial_start = params.trial_start;
trial_max = params.trial_max; %determines which trial number the optimise function begins iterating at. This
%should always be the same as trial_max unless using the Model_performance function.

%the learning rate that fminsearch begins from
alpha_it = [0.001; 0.0001];

%The deca_step is used to only perform e.g. 10,20,30,40 trials rather than
%every iteration of 10-40 if the user sets the deca_operator to 1. This
%will only occur when using optimise with the Model_performance function.
deca_step = 1;
if params.deca_operator == 1
        deca_step = 10;
        trial_start = trial_start*10;
        trial_max = trial_max*10;
end

%keeps track of which trial iteration, regardless of whether deca_operator
%used
trial_counter = 0;

%Iterate through loop, on each of which the number of simulation
%trials increases
for trials = trial_start:deca_step:trial_max
    
    %Generate x and y data-set; note first column is for training, second
    %column is for testing
    %This process is split such that dynamic data sets can be generated
    %where the parameters determining the distribution change mid-task
    z_base = [normrnd(mu_z, sqrt(var_z_A), trial_max/2, 2); normrnd(mu_z, sqrt(var_z_B), trial_max/2, 2)];
    x_base = ones(trial_max, 2);
    y_base = ones(trial_max, 2);
    for ii = 1:2
        for kk = 1:(trial_max/2)
            x_base(kk, ii) = normrnd(omega1A*z_base(kk, ii), sqrt(var_x_A));
            y_base(kk, ii) = normrnd(omega2*z_base(kk, ii), sqrt(var_y));
        end
        for kk = ((trial_max/2)+1):trial_max
            x_base(kk, ii) = normrnd(omega1B*z_base(kk, ii), sqrt(var_x_B));
            y_base(kk, ii) = normrnd(omega2*z_base(kk, ii), sqrt(var_y));
        end
    end
    
    trial_counter = trial_counter + 1;
    
    %fminsearch function returns a vector containing the optimal alpha value for the data-set
    %followed by the value of the cost function (the summed squared error) at that optimal alpha value
    [optimal_ExpProb(trial_counter, :), min_cost_ExpProb(trial_counter)] = fminsearch(@(alpha_it)cost_fun_ExpProb(alpha_it, x_base, y_base), ...
        alpha_it);
    [optimal_ExpReg(trial_counter, :), min_cost_ExpReg(trial_counter)] = fminsearch(@(alpha_it)cost_fun_ExpReg(alpha_it, x_base, y_base), ...
        alpha_it);
    [optimal_CovM(trial_counter, :), min_cost_CovM(trial_counter)] = fminsearch(@(alpha_it)cost_fun_CovM(alpha_it, x_base, y_base), ...
        alpha_it);
    [optimal_Common(trial_counter, :), min_cost_Common(trial_counter)] = fminsearch(@(alpha_it)cost_fun_Common(alpha_it, x_base, y_base),...
        alpha_it);
    
    
    %calculates the RMSD associated with the optimal alpha
    %value
    ExpProb_RMSD(trial_counter) = sqrt(min_cost_ExpProb(trial_counter)/trials);
    ExpReg_RMSD(trial_counter) = sqrt(min_cost_ExpReg(trial_counter)/trials);
    CovM_RMSD(trial_counter) = sqrt(min_cost_CovM(trial_counter)/trials);
    Common_RMSD(trial_counter) = sqrt(min_cost_Common(trial_counter)/trials);
end

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_ExpProb(alpha_it, x_base, y_base)

%Run simulation to generate trained parameters for model.
[~, ExpProb_mu_estimate, var_x_estimate, var_y_estimate, omega_estimate] ...
    = ExpProb(alpha_it, x_base(:, 1), y_base(:, 1));
ExpProb_mu_estimate = ExpProb_mu_estimate(end);
var_x_estimate = var_x_estimate(end);
var_y_estimate = var_y_estimate(end);
omega_estimate = omega_estimate(end);

%Use generated parameters to generate x-estimates.
x_sim = (ExpProb_mu_estimate * var_y_estimate + omega_estimate * y_base(:, 2) * var_x_estimate)...
        /(var_y_estimate + (omega_estimate^2) * var_x_estimate);

%Calculate cost (trained model estimates of testing data set vs. testing
%data set)
cost = sum(((x_sim - x_base(:, 2)).^2));

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_ExpReg(alpha_it, x_base, y_base)

%Run simulation to generate trained parameters for model.
[~, a_estimate, b_estimate] ...
    = ExpReg(alpha_it, x_base(:, 1), y_base(:, 1));
a_estimate = a_estimate(end);
b_estimate = b_estimate(end);

%Use generated parameters to generate x-estimates.
x_sim = a_estimate * y_base(:, 2) + b_estimate;

%Calculate cost (trained model estimates of testing data set vs. testing
%data set)
cost = sum(((x_sim - x_base(:, 2)).^2));

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_CovM(alpha_it, x_base, y_base)

%Run simulation to generate trained parameters for model.
[~, CovM_mu_estimate, covar] ...
    = CovM(alpha_it, x_base(:, 1), y_base(:, 1)); 
CovM_mu_estimate = CovM_mu_estimate(:, end);
covar_inv = inv(covar); %calcuate inverse covariance matrix

%Use generated parameters to generate x-estimates.
x_sim = (covar_inv(1,1) * CovM_mu_estimate(1) + covar_inv(1,2) * CovM_mu_estimate(2)...
        - covar_inv(1,2) * y_base(:, 2)) / covar_inv(1,1);

%Calculate cost (trained model estimates of testing data set vs. testing
%data set)
cost = sum(((x_sim - x_base(:, 2)).^2));

%Returns the cost associated with a particular learning rate
function [cost] = cost_fun_Common(alpha_it, x_base, y_base)

%Run simulation to generate trained parameters for model.
[~, Com_mu_z_estimate, var_z_estimate, Com_mu_estimate, var_estimate, Com_omega_estimate] ...
    = Common(alpha_it, x_base(:, 1), y_base(:, 1));
Com_mu_z_estimate = Com_mu_z_estimate(end);
var_z_estimate = var_z_estimate(end);
Com_mu_estimate = Com_mu_estimate(:, end);
var_estimate = var_estimate(:, end);
Com_omega_estimate = Com_omega_estimate(:, end);

%Use generated parameters to generate x-estimates.
x_sim =  Com_mu_estimate(1) + Com_omega_estimate(1) *...
        (Com_mu_z_estimate * var_estimate(2) + (y_base(:, 2) - Com_mu_estimate(2)) * Com_omega_estimate(2) * var_z_estimate)...
        / (var_estimate(2) + var_z_estimate * (Com_omega_estimate(2)^2));

%Calculate cost (trained model estimates of testing data set vs. testing
%data set)
cost = sum(((x_sim - x_base(:, 2)).^2));
