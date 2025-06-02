import numpy as np
import scipy.io # For potential future use with specific .mat trajectories
from robot_utils import inverse_kinematics, generate_cubic_bridge_trajectory, forward_kinematics, normalize_python
# forward_dynamics is not directly used here for generation, but for calculating tau (inverse dynamics)

def generate_desired_trajectory(traj_type, bridge_type,
                                time_infor, control_infor_start,
                                robot_properties, traj_frequency, dt):
    """
    Generates a desired robot trajectory (q, q_dot, q_ddot, tau) based on Cartesian path.
    """
    # --- Initial Setup ---
    l1 = robot_properties['l1']
    l2 = robot_properties['l2']
    # Other properties (m1, m2, lc1, lc2, I1, I2) will be used for tau calculation

    val_length = time_infor['val_length'] # Expected number of samples for the main part

    q_start = np.asarray(control_infor_start['q_control_start'])
    qdt_start = np.asarray(control_infor_start['qdt_control_start'])

    x_start, y_start = forward_kinematics(q_start[0], q_start[1], l1, l2)

    # Generate a time vector long enough for Cartesian path generation
    # MATLAB's t for x,y generation seems to be 0:dt:val_length, then x,y are used up to 2*val_length+1
    # Let's create t for a bit more than 2*val_length samples
    # Max time needed for path generation before slicing
    # The actual number of points will be determined by val_length + bridge, or just val_length
    # For now, generate enough points for initial x,y generation
    # The final length will be closer to val_length after bridging and IK.
    # Let's generate for roughly 2.5 * val_length * dt duration
    t_duration_raw = (val_length * 2.5) * dt
    t_raw = np.arange(0, t_duration_raw, dt)

    # --- Trajectory Generation (Cartesian x, y) ---
    x_raw = np.zeros_like(t_raw)
    y_raw = np.zeros_like(t_raw)

    if traj_type == 'circle':
        # In MATLAB: x_max = 0.5, y_max = 0.5. Period T = traj_frequency
        # x = x_max * cos(2*pi*t_raw*(1/T))
        # y = y_max * sin(2*pi*t_raw*(1/T))
        x_raw = 0.5 * np.cos(2 * np.pi * t_raw * (1 / traj_frequency))
        y_raw = 0.5 * np.sin(2 * np.pi * t_raw * (1 / traj_frequency))
    elif traj_type == 'infty':
        # In MATLAB: x_max = 0.25, y_max = 0.15. Period T = 2*traj_frequency for x, T for y
        x_raw = 0.25 * np.sin(2 * np.pi * t_raw * (1 / (2 * traj_frequency))) # Note: sin for x
        y_raw = 0.15 * np.sin(2 * np.pi * t_raw * (1 / traj_frequency))      # Note: sin for y
    elif traj_type == 'lorenz':
        try:
            lorenz_data = scipy.io.loadmat('./read_data/lorenz.mat')
            ts_train = lorenz_data['ts_train']

            # val_length is from time_infor['val_length']
            # MATLAB: lorenz_xy=ts_train(1000:1000+round(val_length*2.1), 1:2);
            # Python 0-indexed for start_idx
            start_idx = 1000 - 1
            # Calculate end_idx based on the original val_length expected for the trajectory part
            # not the t_duration_raw or len(t_raw) which is for initial generation.
            # The val_length here is the one passed into the function via time_infor.
            # This is used to determine how much of lorenz data to potentially use.
            num_points_to_extract = int(round(val_length * 2.1)) # Use the val_length from input 'time_infor'

            end_idx = start_idx + num_points_to_extract

            if end_idx > ts_train.shape[0]:
                print(f"Warning: Lorenz data not long enough for desired extraction. End index {end_idx} > data length {ts_train.shape[0]}. Truncating.")
                end_idx = ts_train.shape[0]
            if start_idx >= ts_train.shape[0]:
                print(f"Error: Lorenz data start index {start_idx} is out of bounds for data length {ts_train.shape[0]}.")
                raise IndexError("Lorenz data start index out of bounds.")

            lorenz_xy = ts_train[start_idx:end_idx, 0:2]

            if lorenz_xy.shape[0] < 10: # Arbitrary small number to indicate real data problem
                print("Error: Extracted Lorenz data is too short. Using zeros.")
                x_raw = np.zeros(len(t_raw)) # Fallback to prevent downstream errors with x_raw length
                y_raw = np.zeros(len(t_raw))
            else:
                # Normalize and use a portion of Lorenz data, then potentially repeat or ensure x_raw/y_raw are filled
                # For now, let's assume lorenz_xy is long enough or we need to make x_raw, y_raw match its length.
                # The rest of the code expects x_raw, y_raw to be of len(t_raw).
                # We should make lorenz_x_raw, lorenz_y_raw of a length that can be used to determine add_id.
                # The subsequent logic will handle bridge and final length.
                # So, lorenz_x_raw should be long enough for add_id search and bridge target.
                # Let's make it at least val_length + buffer long, by tiling if necessary.

                lorenz_x_normalized = normalize_python(lorenz_xy[:, 0], [-0.5, 0.5])
                lorenz_y_normalized = normalize_python(lorenz_xy[:, 1], [-0.5, 0.5])
                lorenz_y_normalized = lorenz_y_normalized - 0.3 # As in MATLAB

                # Ensure x_raw and y_raw are populated and have the length of t_raw
                # by tiling the extracted lorenz data if it's shorter than t_raw.
                num_tiles_x = (len(t_raw) + len(lorenz_x_normalized) - 1) // len(lorenz_x_normalized)
                num_tiles_y = (len(t_raw) + len(lorenz_y_normalized) - 1) // len(lorenz_y_normalized)

                x_raw = np.tile(lorenz_x_normalized, num_tiles_x)[:len(t_raw)]
                y_raw = np.tile(lorenz_y_normalized, num_tiles_y)[:len(t_raw)]

        except FileNotFoundError:
            print("Error: ./read_data/lorenz.mat not found. Using zeros for Lorenz trajectory.")
            x_raw = np.zeros(len(t_raw))
            y_raw = np.zeros(len(t_raw))
        except KeyError:
            print("Error: Key 'ts_train' not found in lorenz.mat. Using zeros for Lorenz trajectory.")
            x_raw = np.zeros(len(t_raw))
            y_raw = np.zeros(len(t_raw))
        except IndexError as e: # Catch specific index error from start_idx
             print(f"IndexError during Lorenz data processing: {e}. Using zeros for Lorenz trajectory.")
             x_raw = np.zeros(len(t_raw))
             y_raw = np.zeros(len(t_raw))
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    # Initial slicing: ensure we have enough points for finding add_id and a bit beyond.
    # MATLAB uses x(1:val_length) for finding add_id.
    # Let's ensure we have at least val_length points, plus some buffer for add_id+1 etc.
    # Max index needed for raw x,y could be around val_length + few points for bridge target velocity.
    # The final trajectory will be constructed to be around val_length.
    # Slice to a working length for now, e.g., up to val_length + some_buffer
    # The MATLAB code implies that x, y are generated, then a bridge is prepended,
    # and the total length is then managed.
    # Let's keep x_raw, y_raw relatively long for now. The final length will be determined later.
    # Max needed for add_id search and next point: val_length + 1
    search_len = val_length + 2 # Ensure x_raw[add_id+1] is valid
    if len(x_raw) < search_len:
        # This should not happen with t_duration_raw = (val_length * 2.5) * dt
        print(f"Warning: Raw trajectory too short (len {len(x_raw)} vs needed {search_len}). May fail.")
        # Potentially extend t_raw and regenerate x_raw, y_raw if this happens.

    # --- Bridge Calculation ---
    # 1. Find the closest point on the raw (x,y) trajectory to (x_start, y_start)
    # Search within the first `val_length` points of the generated (x_raw, y_raw)
    distances = np.sqrt((x_start - x_raw[:val_length])**2 + (y_start - y_raw[:val_length])**2)
    add_id = np.argmin(distances) # Index in x_raw, y_raw

    x_bridge_target = x_raw[add_id]
    y_bridge_target = y_raw[add_id]

    # 2. Determine bridge_duration
    bridge_len = np.sqrt((x_bridge_target - x_start)**2 + (y_bridge_target - y_start)**2)
    # MATLAB: bridge_time = round(bridge_len * 1/dt). This is number of samples.
    # bridge_duration_in_time = bridge_time_samples * dt
    # If speed is 1 unit/sec: bridge_duration_in_time = bridge_len / 1.0
    # The MATLAB variable bridge_time seems to be duration in actual time units, not samples.
    # Let's assume speed = 1 unit/sec for calculating duration.
    bridge_duration_in_time = bridge_len / 1.0 # Assuming speed of 1 unit/sec

    if add_id + 1 >= len(x_raw):
        print("Error: Cannot get x_target_next, add_id is too close to end of raw trajectory.")
        # This might happen if val_length is very large compared to t_raw generation.
        # Or if x_start is very far from the generated path.
        # Fallback: use target point itself for next point (zero target velocity)
        x_target_next = x_bridge_target
        y_target_next = y_bridge_target
        print("Warning: Using zero target velocity for bridge due to add_id at end of raw trajectory.")
    else:
        x_target_next = x_raw[add_id+1]
        y_target_next = y_raw[add_id+1]

    q_bridge, xy_bridge = None, None
    if bridge_duration_in_time > dt: # Only generate bridge if it's longer than a timestep
        q_bridge, xy_bridge = generate_cubic_bridge_trajectory(
            q_start, qdt_start,
            x_bridge_target, y_bridge_target,
            x_target_next, y_target_next,
            l1, l2, dt, bridge_duration_in_time
        )
        if q_bridge is None: # IK failed during bridge generation
            print("Bridge generation failed (IK issue). Proceeding without bridge.")
    else:
        print("Bridge duration too short, skipping bridge generation.")

    if xy_bridge is not None and len(xy_bridge) > 1:
        # MATLAB uses truePosition(:, 2:end-1) for the bridge part in final traj.
        # This means it excludes the very first point (q_start) and the very last point (connection point)
        # of the bridge from the bridge-specific part of the trajectory.
        # The first point of xy_bridge is x_start, y_start.
        # The last point of xy_bridge is x_bridge_target, y_bridge_target.
        # x_final starts with xy_bridge[1:-1,0] then x_raw[add_id+1:].
        # This seems to skip the connection point x_raw[add_id].
        # Let's verify MATLAB: truePosition is bridge. x_final = [bridge_x(2:end-1), regular_x(add_id:end)]
        # This means bridge's first point (actual start) and last point (connection to main traj) are excluded
        # from the bridge segment, and the main traj starts from add_id.
        # This would mean the connection point is duplicated if not handled.
        # If xy_bridge is (start, p1, p2, ..., connect_pt), and x_raw is (..., connect_pt, p_next, ...)
        # then xy_bridge[1:-1,0] is (p1,p2,...)
        # and x_raw[add_id,:] is (connect_pt, p_next)
        # So, np.concatenate((xy_bridge[1:-1,0], x_raw[add_id:,0])) seems correct.
        # The length of xy_bridge[1:-1,0] is num_bridge_samples - 2.
        x_final = np.concatenate((xy_bridge[1:-1, 0], x_raw[add_id:]))
        y_final = np.concatenate((xy_bridge[1:-1, 1], y_raw[add_id:]))
    else: # No bridge, or bridge is too short
        x_final = x_raw[add_id:]
        y_final = y_raw[add_id:]
        # We need to ensure x_final starts from x_start if no bridge is used.
        # If add_id was found based on x_start, then x_raw[add_id] is the closest point.
        # If bridge is skipped, the trajectory should ideally still start from x_start.
        # This means the "raw" trajectory effectively starts from x_raw[add_id].
        # This part needs to align with how control_infor_start (q_start, qdt_start) is handled.
        # For now, this implies a jump if x_start != x_raw[add_id].
        # TODO: Revisit this logic if no bridge. For now, proceed.
        print("No bridge or short bridge: x_final starts from x_raw[add_id].")


    # Ensure final trajectory is at least val_length long.
    # MATLAB's final trueCtrlTraj is val_length long.
    # If len(x_final) < val_length, it might need padding or error.
    # If len(x_final) > val_length, it's truncated.
    if len(x_final) >= val_length:
        x_control_cartesian = x_final[:val_length]
        y_control_cartesian = y_final[:val_length]
    else:
        print(f"Warning: Final trajectory length {len(x_final)} is less than val_length {val_length}.")
        # Pad with the last point if necessary, though this indicates an issue.
        x_control_cartesian = np.pad(x_final, (0, val_length - len(x_final)), 'edge')
        y_control_cartesian = np.pad(y_final, (0, val_length - len(y_final)), 'edge')
        print(f"Padded to {len(x_control_cartesian)}.")

    final_num_points = len(x_control_cartesian)

    # --- Inverse Kinematics ---
    q_control = np.zeros((final_num_points, 2))
    # For IK, q2 sign consistency is important.
    # Use q_start[1] (second joint angle) to determine initial preference for q2 sign.
    # Then, maintain consistency by choosing next q2 to be close to previous q2.

    prev_q2 = q_start[1] # Initial reference for q2 sign

    for i in range(final_num_points):
        q1_i_raw, q2_i_raw = inverse_kinematics(x_control_cartesian[i], y_control_cartesian[i], l1, l2)

        if q1_i_raw is None: # Unreachable point
            print(f"Warning: Point ({x_control_cartesian[i]:.2f}, {y_control_cartesian[i]:.2f}) at index {i} is unreachable.")
            q_control[i,:] = q_control[i-1,:] if i > 0 else q_start
            # prev_q2 remains from the last successful IK
            continue

        # Simplified IK choice: Use the solution from inverse_kinematics.
        # More advanced: Choose q2_i such that it's closer to prev_q2, or matches sign of prev_q2.
        # Our current IK returns q2_raw in [0, pi].
        # If prev_q2 was negative, we might prefer the negative solution for q2_i_raw.
        chosen_q1 = q1_i_raw
        chosen_q2 = q2_i_raw

        if i == 0: # For the first point after the bridge (or the very first point if no bridge)
            # Choose q2 that matches sign of q_start[1] if possible
            if q_start[1] < 0 and chosen_q2 > 1e-6 : # if q_start[1] is negative and chosen_q2 is positive
                # Recalculate q1 with -chosen_q2
                q2_alt = -chosen_q2
                q1_alt = np.arctan2(y_control_cartesian[i], x_control_cartesian[i]) - \
                         np.arctan2(l2 * np.sin(q2_alt), l1 + l2 * np.cos(q2_alt))
                chosen_q1 = q1_alt
                chosen_q2 = q2_alt
        else: # For subsequent points, maintain continuity with prev_q2
            # If chosen_q2 (positive) has different sign than prev_q2 (negative)
            if prev_q2 < -1e-6 and chosen_q2 > 1e-6:
                 # Consider the alternative solution for q2 (-chosen_q2)
                 q2_alt = -chosen_q2
                 # If -chosen_q2 is closer to prev_q2 than chosen_q2 is
                 if np.abs(q2_alt - prev_q2) < np.abs(chosen_q2 - prev_q2):
                    q1_alt = np.arctan2(y_control_cartesian[i], x_control_cartesian[i]) - \
                             np.arctan2(l2 * np.sin(q2_alt), l1 + l2 * np.cos(q2_alt))
                    chosen_q1 = q1_alt
                    chosen_q2 = q2_alt
            # Also handle q1 unwrapping (simplified: ensure smallest angle change)
            if i > 0: # Ensure q_control[i-1,0] is valid
                current_q1 = q_control[i-1, 0]
                # Check if chosen_q1 needs to be adjusted by +/- 2*pi to be closer to current_q1
                # This basic unwrapping might not be perfect but is a start.
                if np.abs(chosen_q1 - current_q1) > np.pi:
                    if chosen_q1 > current_q1:
                        chosen_q1 -= 2 * np.pi
                    else:
                        chosen_q1 += 2 * np.pi
                # Second check if it overshot
                if np.abs(chosen_q1 - current_q1) > np.pi: # Should not happen with correct logic above
                     # This case means that even after one flip, it's still far.
                     # This can happen if the jump is very large, close to multiples of 2pi.
                     # A more robust unwrapper might be needed, but this covers simple cases.
                     pass # Stick with the current chosen_q1 after one attempt

        q_control[i,:] = [chosen_q1, chosen_q2]
        prev_q2 = chosen_q2


    # --- Derivatives and Torques (Inverse Dynamics) ---
    qdt_control = np.gradient(q_control, dt, axis=0, edge_order=2)
    q2dt_control = np.gradient(qdt_control, dt, axis=0, edge_order=2)

    tau_control = np.zeros_like(q_control)

    m1 = robot_properties['m1']; m2 = robot_properties['m2']
    lc1 = robot_properties['lc1']; lc2 = robot_properties['lc2']
    I1 = robot_properties['I1']; I2 = robot_properties['I2']

    for ii in range(final_num_points):
        q1_curr = q_control[ii, 0]
        q2_curr = q_control[ii, 1]
        qdt1_curr = qdt_control[ii, 0]
        qdt2_curr = qdt_control[ii, 1]
        q2dt1_curr = q2dt_control[ii, 0] # Error: Should be q2dt_control[ii,0]
        q2dt2_curr = q2dt_control[ii, 1] # Error: Should be q2dt_control[ii,1]

        # H-matrix components
        H11 = m1*lc1**2 + I1 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q2_curr)) + I2
        H12 = m2*l1*lc2*np.cos(q2_curr) + m2*lc2**2 + I2
        H21 = H12
        H22 = m2*lc2**2 + I2

        # Coriolis/centrifugal terms (part_1 corresponds to C_qd_1, part_2 to C_qd_2 from forward_dynamics)
        h_term_factor = m2*l1*lc2*np.sin(q2_curr)
        C_qd_1 = -h_term_factor * qdt2_curr * qdt1_curr - h_term_factor * (qdt1_curr + qdt2_curr) * qdt2_curr
        C_qd_2 =  h_term_factor * qdt1_curr**2

        # Tau = H*q2dt + C*qdt (where C*qdt are the part_1, part_2 terms)
        # Corrected lines for q2dt usage:
        tau_control[ii, 0] = H11*q2dt_control[ii,0] + H12*q2dt_control[ii,1] + C_qd_1
        tau_control[ii, 1] = H21*q2dt_control[ii,0] + H22*q2dt_control[ii,1] + C_qd_2

    return {
        'q_control': q_control,
        'qdt_control': qdt_control,
        'q2dt_control': q2dt_control,
        'tau_control': tau_control,
        'x_control_cartesian': x_control_cartesian,
        'y_control_cartesian': y_control_cartesian,
        'final_val_length': final_num_points
    }
