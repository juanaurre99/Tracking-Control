% run_bayesopt_double_arm.m
% Bayesian optimization for double-arm reservoir control

clear; clc;

%% --- 1. Generate and Save Training Data Once (if not already saved) ---

data_file = 'fixed_robot_training_data.mat';
if ~isfile(data_file)
    fprintf('Generating training data and saving to %s...\n', data_file);

    dt = 0.01;
    properties = [1,1,0.5,0.5,0.25,0.25,0.03,0.03];
    noise_level = 2.0e-2;
    input_infor = {'xy', 'qdt'};
    dim_in = numel(input_infor) * 4;
    dim_out = 2;
    reset_t = 80;

    section_len    = round(reset_t/dt);
    washup_length  = round(1002/dt);
    train_length   = round(1000/dt) + round(5/dt);
    val_length     = round(500/dt);
    time_length    = train_length + 2*val_length + 3*washup_length + 100;

    time_infor = struct('section_len', section_len, 'washup_length', washup_length, ...
        'train_length', train_length, 'val_length', val_length, 'time_length', time_length);

    % Replace this with your own robot_data_generator function as needed:
    [xy, q, qdt, q2dt, tau] = robot_data_generator(time_infor, noise_level, dt, properties);

    xy   = xy(washup_length:end, :);
    q    = q(washup_length:end, :);
    qdt  = qdt(washup_length:end, :);
    q2dt = q2dt(washup_length:end, :);
    tau  = tau(washup_length:end, :);

    save(data_file, 'xy', 'q', 'qdt', 'q2dt', 'tau', ...
        'dt', 'properties', 'input_infor', 'dim_in', 'dim_out', 'reset_t', 'noise_level', 'time_infor');
else
    fprintf('Training data file %s already exists. Loading in objective...\n', data_file);
end

%% --- 2. Define Hyperparameter Search Space ---
vars = [ ...
    optimizableVariable('eig_rho',      [0.5, 1.5]), ...
    optimizableVariable('W_in_a',       [0.5, 1.5]), ...
    optimizableVariable('alpha',        [0.5, 1.0]), ...
    optimizableVariable('log10_beta',   [-4, -1]), ...
    optimizableVariable('k_scaled',     [40, 160]), ...
    optimizableVariable('kb',           [0.5, 3.0]) ...
];

%% --- 3. Objective Wrapper for bayesopt ---
obj_wrapper = @(tbl) objective_double_arm( ...
    [tbl.eig_rho, tbl.W_in_a, tbl.alpha, tbl.log10_beta, tbl.k_scaled, tbl.kb] ...
);

%% --- 4. Run Bayesian Optimization ---
results = bayesopt(obj_wrapper, vars, ...
    'MaxObjectiveEvaluations', 8, ...   % Increase for real runs (e.g., 30-100+)
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'PlotFcn', {@plotObjectiveModel, @plotMinObjective});

%% --- 5. Show Best Hyperparameters and Score ---
fprintf('\nBest hyperparameters found:\n');
disp(results.XAtMinObjective)
fprintf('Best (lowest) mean RMSE: %.5f\n', results.MinObjective);

% Optionally load models (adjust names if you save them differently)
if isfile('first_model.mat'), load first_model.mat, end
if isfile('best_model.mat'), load best_model.mat, end

% Choose trajectory index (e.g., 2 for 'circle')
traj_idx = 2; % 1=lorenz, 2=circle, 3=mg17, 4=infty

