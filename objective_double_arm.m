function a_rmse = objective_double_arm(x)
% Objective function for hyperparameter optimization of double-arm reservoir control.
% Loads pre-generated training data for reproducibility.
% x: [eig_rho, W_in_a, alpha, log10_beta, k_scaled, kb]

    persistent first_model_vars best_rmse call_count ...
               xy q qdt q2dt tau dt properties input_infor dim_in dim_out reset_t noise_level time_infor

    % Load training data once per session
    if isempty(xy)
        S = load('fixed_robot_training_data.mat');
        xy         = S.xy;
        q          = S.q;
        qdt        = S.qdt;
        q2dt       = S.q2dt;
        tau        = S.tau;
        dt         = S.dt;
        properties = S.properties;
        input_infor = S.input_infor;
        dim_in     = S.dim_in;
        dim_out    = S.dim_out;
        reset_t    = S.reset_t;
        noise_level = S.noise_level;
        time_infor = S.time_infor;
    end

    if isempty(call_count)
        call_count = 1;
    else
        call_count = call_count + 1;
    end

    n = 200; % Reservoir size

    % Hyperparameters
    eig_rho  = x(1);
    W_in_a   = x(2);
    alpha    = x(3);
    beta     = 10^x(4);
    k        = round(x(5) / 200 * n);
    kb       = x(6);

    % Reservoir creation
    W_in = W_in_a * (2*rand(n, dim_in)-1);
    res_net = sprandsym(n, k/n);
    eig_D = eigs(res_net,1);
    res_net = (eig_rho/(abs(eig_D))) .* res_net;
    res_net = full(res_net);

    res_infor = struct('W_in', W_in, 'res_net', res_net, 'alpha', alpha, ...
                       'kb', kb, 'beta', beta, 'n', n);

    data_reservoir = struct('xy', xy, 'q', q, 'qdt', qdt, 'q2dt', q2dt, 'tau', tau);

    % Train
    [Wout, r_end] = func_reservoir_train(data_reservoir, time_infor, input_infor, res_infor, dim_in, dim_out);

    % Default start_info, failure, blur, traj_frequency
    start_info = [];
    failure.type = 'none';
    blur.blur = 0;
    traj_frequency = 1; % Set as needed

    % Validation params
    rmse_start_time = 200000;
    rmse_end_time   = 300000-100;
    time_infor.val_length = 300000;

    % Validation: for each trajectory
    traj_types = {'lorenz', 'circle', 'mg17', 'infty'};
    bridge_type = 'cubic';
    rmse_vec = zeros(1, numel(traj_types));
    traj_results = cell(1, numel(traj_types)); % For storing trajectory data (optional)

    for i = 1:numel(traj_types)
        traj_type = traj_types{i};
        plot_movie = 0;
        save_rend  = 0;

        [control_infor, output_infor, ~, ~] = func_reservoir_validate( ...
            traj_type, bridge_type, time_infor, input_infor, res_infor, ...
            start_info, properties, dim_in, dim_out, Wout, r_end, dt, ...
            plot_movie, save_rend, failure, blur, traj_frequency);

        data_pred    = output_infor.data_pred;
        data_control = control_infor.data_control;
        rmse_vec(i)  = func_rmse(data_pred, data_control, rmse_start_time, rmse_end_time);

        % Optional: store trajectory results
        traj_results{i} = struct('traj_type', traj_type, 'data_pred', data_pred, 'data_control', data_control);
    end

    a_rmse = mean(rmse_vec);

    % Save first model in current folder, if it's the first call
    if call_count == 1
        if ~exist('./choose_file', 'dir')
            mkdir('./choose_file');
        end
        time_today = datestr(now, 'mmddyyyy');
        filename = ['./choose_file/all_traj_', time_today, '_' num2str(randi(9999)) '_' num2str(randi(9999)) '.mat'];
        save(filename, 'time_infor', 'input_infor', 'res_infor', 'properties', ...
            'dim_in', 'dim_out', 'Wout', 'r_end', 'dt', 'reset_t', 'noise_level', 'a_rmse');
    end

    % Save best model (lowest RMSE so far) in bayesian_opt folder
    if isempty(best_rmse) || a_rmse < best_rmse
        best_rmse = a_rmse;
        if ~exist('./bayesian_opt', 'dir')
            mkdir('./bayesian_opt');
        end
        time_today = datestr(now, 'mmddyyyy');
        filename = ['./bayesian_opt/all_traj_', time_today, '_' num2str(randi(9999)) '_' num2str(randi(9999)) '.mat'];
        save(filename, 'time_infor', 'input_infor', 'res_infor', 'properties', ...
            'dim_in', 'dim_out', 'Wout', 'r_end', 'dt', 'reset_t', 'noise_level', 'a_rmse');
    end

end
