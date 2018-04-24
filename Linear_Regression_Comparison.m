function Linear_Regression_Comparison

%v3.0

%This function trains each of the models on a training data-set. 
%It then tests their performance, comparing their
%performance to that of a standard linear regression.

%Input parameters
params = struct;
params.mu_z = 5; %mean of the base distribution
params.var_z_A = 1; %variance of base distribution
params.var_z_B = 1; %variance of the z varaible
params.var_x_A = 0; %variance of the x variable
params.var_x_B = 0;
params.var_y = 1; %variance of the y variable
params.omega1A = 1; %linear transformation variable
params.omega1B = 1;
params.omega2 = 1; %second linear transformation variable
params.deca_operator = 0;

params.trial_max = 5000; %number of trials simulated training and testing
params.trial_start = params.trial_max; %determines which trial number the optimise function begins iterating at


%Note columns in base distributions correspond to training data (1) and
%testing data (2)
z_base = [normrnd(params.mu_z, sqrt(params.var_z_A), params.trial_max/2, 2); ...
    normrnd(params.mu_z, sqrt(params.var_z_B), params.trial_max/2, 2)];
x_base = ones(params.trial_max, 2);
y_base = ones(params.trial_max, 2);

for ii = 1:2
    for kk = 1:(params.trial_max/2)
        x_base(kk, ii) = normrnd(params.omega1A*z_base(kk, ii), sqrt(params.var_x_A));
        y_base(kk, ii) = normrnd(params.omega2*z_base(kk, ii), sqrt(params.var_y));
    end
    for kk = ((params.trial_max/2)+1):params.trial_max
        x_base(kk, ii) = normrnd(params.omega1B*z_base(kk, ii), sqrt(params.var_x_B));
        y_base(kk, ii) = normrnd(params.omega2*z_base(kk, ii), sqrt(params.var_y));
    end
end

%identify optimal learning rates using the same probability distrubtion
[optimal_ExpProb, optimal_ExpReg, optimal_CovM, optimal_Common,...
        ExpProb_RMSD, ExpReg_RMSD, CovM_RMSD, Common_RMSD]...
       = Optimise(params);

%generate model parameters on training dataset using optimal learning rate
[~, ExpProb_mu_estimate, var_x_estimate, var_y_estimate, omega_estimate] ...
    = ExpProb(optimal_ExpProb, x_base(:, 1), y_base(:, 1));
ExpProb_mu_estimate = ExpProb_mu_estimate(end);
var_x_estimate = var_x_estimate(end);
var_y_estimate = var_y_estimate(end);
omega_estimate = omega_estimate(end);

[~, a_estimate, b_estimate] ...
    = ExpReg(optimal_ExpReg, x_base(:, 1), y_base(:, 1));
a_estimate = a_estimate(end);
b_estimate = b_estimate(end);

[~, CovM_mu_estimate, covar] ...
    = CovM(optimal_CovM, x_base(:, 1), y_base(:, 1));
CovM_mu_estimate = CovM_mu_estimate(:, end);
covar_inv = inv(covar);

[~, Com_mu_z_estimate, var_z_estimate, Com_mu_estimate, var_estimate, Com_omega_estimate] ...
    = Common(optimal_Common, x_base(:, 1), y_base(:, 1));
Com_mu_z_estimate = Com_mu_z_estimate(end);
var_z_estimate = var_z_estimate(end);
Com_mu_estimate = Com_mu_estimate(:, end);
var_estimate = var_estimate(:, end);
Com_omega_estimate = Com_omega_estimate(:, end);


%generate x-estimates from the testing data, when models are
%*not* able to update their parameters
x_estimate_ExpProb = (ExpProb_mu_estimate * var_y_estimate + omega_estimate * y_base(:, 2) * var_x_estimate)...
        /(var_y_estimate + (omega_estimate^2) * var_x_estimate);
    
x_estimate_ExpReg = a_estimate * y_base(:, 2) + b_estimate;

x_estimate_CovM = (covar_inv(1,1) * CovM_mu_estimate(1) + covar_inv(1,2) * CovM_mu_estimate(2)...
        - covar_inv(1,2) * y_base(:, 2)) / covar_inv(1,1);
    
x_estimate_Common =  Com_mu_estimate(1) + Com_omega_estimate(1) *...
        (Com_mu_z_estimate * var_estimate(2) + (y_base(:, 2) - Com_mu_estimate(2)) * Com_omega_estimate(2) * var_z_estimate)...
        / (var_estimate(2) + var_z_estimate * (Com_omega_estimate(2)^2));
    
    
%calculate and RMSD for each model on the testing set
ExpProb_RMSD = sqrt(sum(((x_estimate_ExpProb - x_base(:, 2)).^2))/length(x_base(:, 2)));
ExpReg_RMSD = sqrt(sum(((x_estimate_ExpReg - x_base(:, 2)).^2))/length(x_base(:, 2)));
CovM_RMSD = sqrt(sum(((x_estimate_CovM - x_base(:, 2)).^2))/length(x_base(:, 2)));
Common_RMSD = sqrt(sum(((x_estimate_Common - x_base(:, 2)).^2))/length(x_base(:, 2)));


%calculates slope (beta) and intercept (alpha) of a simple regression line
%with which to compare models
regression_cov = cov(x_base(:, 2), y_base(:, 2));
beta = regression_cov(1,2)/regression_cov(2,2);
alpha = mean(x_base(:, 2)) - beta*mean(y_base(:, 2));


figure('position', [0, 0, 1400, 600])
subplot(1, 2, 1)
fplot(@(y) (ExpProb_mu_estimate * var_y_estimate + omega_estimate * y * var_x_estimate)...
    /(var_y_estimate + (omega_estimate^2) * var_x_estimate), 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2); %generative ExpProb model line
hold on
fplot(@(y) a_estimate * y + b_estimate, 'Color', [0.6350, 0.0780, 0.1840], 'LineWidth', 2) %Rescorla-Wagner model line
fplot(@(y) alpha + beta*y, '--k', 'LineWidth', 2) %simple linear regression line
scatter(y_base(1:200, 2), x_base(1:200, 2), 'k')
legend('Generative Model', 'Rescorla-Wagner Model', 'Simple Linear Regression')
ylabel('x-output')
xlabel('y-input')
axis([0 10 0 10])
title('Model Estimates On Testing Data Set')

subplot(1, 2, 2)
fplot(@(y) (covar_inv(1,1) * CovM_mu_estimate(1) + covar_inv(1,2) * CovM_mu_estimate(2)...
        - covar_inv(1,2) * y) / covar_inv(1,1), 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2); %covariance line
hold on
fplot(@(y) Com_mu_estimate(1) + Com_omega_estimate(1) * ...
        (Com_mu_z_estimate * var_estimate(2) + (y - Com_mu_estimate(2)) * Com_omega_estimate(2) * var_z_estimate)...
        / (var_estimate(2) + var_z_estimate * (Com_omega_estimate(2)^2)), 'Color', [0.3010, 0.7450, 0.9330], 'LineWidth', 2); %common variable line
fplot(@(y) alpha + beta*y, '--k', 'LineWidth', 2) %simple linear regression line
scatter(y_base(1:200, 2), x_base(1:200, 2), 'k')
legend('Covariance Model', 'Common Variable', 'Simple Linear Regression')
ylabel('x-output')
xlabel('y-input')
axis([0 10 0 10])
title('Model Estimates On Testing Data Set')


