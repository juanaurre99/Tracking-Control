import numpy as np
from robot_utils import forward_dynamics, forward_kinematics # Added forward_kinematics

def run_reservoir_validation(desired_traj_data, model_params, robot_properties, sim_params, input_infor_setup):
    """
    Runs the reservoir validation simulation.
    """
    # --- Initial Setup ---
    # Model parameters
    W_in = model_params['W_in']
    res_net = model_params['res_net']
    Wout = model_params['Wout']
    alpha = model_params['alpha']
    kb = model_params['kb']
    n_reservoir = int(model_params['n']) # Ensure n is integer for reservoir size
    dt = model_params['dt']
    dim_in = int(model_params['dim_in'])

    # Robot physical properties
    l1 = robot_properties['l1']
    l2 = robot_properties['l2']
    m1 = robot_properties['m1']; m2 = robot_properties['m2']
    lc1 = robot_properties['lc1']; lc2 = robot_properties['lc2']
    I1 = robot_properties['I1']; I2 = robot_properties['I2']

    # Desired trajectory data
    q_control = desired_traj_data['q_control']
    qdt_control = desired_traj_data['qdt_control']
    # data_control is [x_control_cartesian, y_control_cartesian]
    # Need to handle if it's passed as two separate arrays or one combined array
    if 'data_control' in desired_traj_data:
        data_control_cartesian = desired_traj_data['data_control'] # Assuming shape (val_length, 2)
    elif 'x_control_cartesian' in desired_traj_data and 'y_control_cartesian' in desired_traj_data:
        data_control_cartesian = np.column_stack((
            desired_traj_data['x_control_cartesian'],
            desired_traj_data['y_control_cartesian']
        ))
    else:
        raise ValueError("Missing 'data_control' or 'x/y_control_cartesian' in desired_traj_data")

    val_length = desired_traj_data['final_val_length']

    # Initialize state variables
    q_pred = np.zeros((val_length, 2))
    qdt_pred = np.zeros((val_length, 2))
    q2dt_pred = np.zeros((val_length, 2)) # Stores actual q2dt experienced by the robot
    tau_pred = np.zeros((val_length, 2))  # Stores the torque *applied* to the robot (after limiting)
    data_pred_cartesian = np.zeros((val_length, 2)) # Stores predicted Cartesian positions

    # Set initial robot state
    q_pred[0, :] = sim_params['start_info_q']
    qdt_pred[0, :] = sim_params['start_info_qdt']

    # Calculate initial Cartesian state from q_pred[0,:]
    x_init_pred, y_init_pred = forward_kinematics(q_pred[0,0], q_pred[0,1], l1, l2)
    data_pred_cartesian[0, :] = [x_init_pred, y_init_pred]

    # Initialize reservoir state r
    if sim_params.get('r_end_initial') is None: # Use .get for safety
        r = np.zeros((n_reservoir, 1))
    else:
        r = sim_params['r_end_initial']

    # Initialize input vector u
    u = np.zeros((dim_in, 1))

    # Torque rate limiter threshold
    taudt_threshold = np.array([-5e-2, 5e-2]) # Store as numpy array

    # Initialize applied torque for t=0 (used for first iteration of limiter)
    # This should be the torque corresponding to the initial state, often from desired trajectory
    # or a specific initial condition.
    tau_pred[0, :] = sim_params.get('start_info_tau', np.zeros(2))


    # --- Main Simulation Loop ---
    # Loop from t_i = 0 to val_length - 2 to predict states for t_i + 1
    # This means q_pred, qdt_pred, etc. will be filled up to index val_length - 1
    for t_i in range(val_length - 1):
        # 1. Construct u (input to reservoir)
        # This depends on input_infor_setup (e.g., ['xy', 'qdt'])
        # u uses current predicted state and next desired state.

        current_actual_cart_x = data_pred_cartesian[t_i, 0]
        current_actual_cart_y = data_pred_cartesian[t_i, 1]
        current_actual_qdt1 = qdt_pred[t_i, 0]
        current_actual_qdt2 = qdt_pred[t_i, 1]

        # Index for "next desired state" from control trajectory
        # MATLAB's time_now+2 (1-based) becomes t_i+1 for 0-based current step t_i,
        # if "next" means one step ahead of current control target.
        # Or if it means two steps ahead of current physical step t_i.
        # The MATLAB code for u: u(3:4)=data_control(time_now+2,:); u(7:8)=qdt_control(time_now+2,:);
        # If time_now is the current physical step, then time_now+2 is two steps ahead.
        # Let's use t_i + 1 as the "next" lookahead for desired states.
        idx_desired_next = min(t_i + 1, val_length - 1)

        u_list = []
        # The exact construction of 'u' must match how the reservoir was trained.
        # Assuming typical setup based on provided snippet:
        # [current_actual_xy, next_desired_xy, current_actual_qdt, next_desired_qdt]
        if 'xy' in input_infor_setup and 'qdt' in input_infor_setup: # Common case
            u_list.extend([current_actual_cart_x, current_actual_cart_y])
            u_list.extend([data_control_cartesian[idx_desired_next, 0], data_control_cartesian[idx_desired_next, 1]])
            u_list.extend([current_actual_qdt1, current_actual_qdt2])
            u_list.extend([qdt_control[idx_desired_next, 0], qdt_control[idx_desired_next, 1]])
        # Add more conditions here if input_infor_setup can vary more
        else:
            # Fallback or error if u construction is not defined for input_infor_setup
            raise NotImplementedError(f"Input vector 'u' construction not defined for input_infor_setup: {input_infor_setup}")

        if len(u_list) != dim_in:
            raise ValueError(f"Constructed u length {len(u_list)} does not match dim_in {dim_in}")
        u = np.array(u_list).reshape(dim_in, 1)

        # 2. Reservoir Update
        r = (1 - alpha) * r + alpha * np.tanh(res_net @ r + W_in @ u + kb * np.ones((n_reservoir, 1)))

        # 3. Output Calculation (Predict torque for current step t_i based on state at t_i)
        r_out = r.copy()
        r_out[1::2] = r_out[1::2]**2 # Square elements at odd indices (1, 3, 5, ...)

        current_tau_prediction_from_reservoir = Wout @ r_out # Shape (2,1)

        # 4. Apply Disturbances/Noise (Skipped for now)

        # 5. Torque Limiting
        # current_tau_prediction_from_reservoir is for the actions to take at step t_i
        # It should be compared against the torque applied at t_i-1, which is tau_pred[t_i-1,:]

        # Let tau_applied_at_previous_step be the reference for limiting
        # For t_i = 0, this is the initial torque (e.g. sim_params.get('start_info_tau'))
        # For t_i > 0, this is tau_pred[t_i-1, :] (the torque applied at the previous step)

        tau_at_prev_step = tau_pred[t_i-1, :] if t_i > 0 else sim_params.get('start_info_tau', np.zeros(2))

        limited_tau_for_t_i = np.zeros(2)
        for joint_idx in range(2):
            delta_tau = current_tau_prediction_from_reservoir[joint_idx,0] - tau_at_prev_step[joint_idx]
            if delta_tau > taudt_threshold[1] * dt:
                limited_tau_for_t_i[joint_idx] = tau_at_prev_step[joint_idx] + taudt_threshold[1] * dt
            elif delta_tau < taudt_threshold[0] * dt:
                limited_tau_for_t_i[joint_idx] = tau_at_prev_step[joint_idx] + taudt_threshold[0] * dt
            else:
                limited_tau_for_t_i[joint_idx] = current_tau_prediction_from_reservoir[joint_idx,0]

        tau_pred[t_i, :] = limited_tau_for_t_i # Store the applied torque for step t_i

        # 6. Robot State Update (using applied torque at t_i)
        # current_q and current_qdt are from q_pred[t_i, :] and qdt_pred[t_i, :]
        current_q_for_dynamics = q_pred[t_i, :].reshape(2,1) # Ensure column vector
        current_qdt_for_dynamics = qdt_pred[t_i, :].reshape(2,1) # Ensure column vector
        applied_tau_for_dynamics = tau_pred[t_i, :].reshape(2,1) # Ensure column vector

        q2dt_val = forward_dynamics(current_q_for_dynamics, current_qdt_for_dynamics, applied_tau_for_dynamics,
                                    m1,m2,l1,l2,lc1,lc2,I1,I2)
        q2dt_pred[t_i, :] = q2dt_val.flatten() # Store actual acceleration experienced at t_i

        # Update next step's q and qdt using dynamics at t_i
        # q_pred[t_i+1], qdt_pred[t_i+1]
        qdt_pred[t_i+1, :] = qdt_pred[t_i, :] + q2dt_pred[t_i, :] * dt
        q_pred[t_i+1, :] = q_pred[t_i, :] + qdt_pred[t_i, :] * dt # Using qdt_pred[t_i] as per MATLAB

        # 7. Calculate Cartesian Prediction for next iteration's 'u' construction
        # This is data_pred_cartesian at t_i+1
        x_p_next, y_p_next = forward_kinematics(q_pred[t_i+1, 0], q_pred[t_i+1, 1], l1, l2)
        data_pred_cartesian[t_i+1, :] = [x_p_next, y_p_next]
        # (Apply measurement noise here if any, for now skip)

    # --- Final r_end --- (Reservoir state after the loop)
    r_end = r

    return {
        'data_pred_cartesian': data_pred_cartesian,
        'q_pred': q_pred,
        'qdt_pred': qdt_pred,
        'q2dt_pred': q2dt_pred,
        'tau_pred': tau_pred, # These are the torques applied to the robot
        'r_end': r_end
    }
