import scipy.io
import numpy as np
import matplotlib.pyplot as plt # Added for plotting

# Assuming robot_utils and trajectory_generator are in the same directory or PYTHONPATH
from robot_utils import forward_kinematics, inverse_kinematics, normalize_python, generate_cubic_bridge_trajectory, forward_dynamics
from trajectory_generator import generate_desired_trajectory
from reservoir_validation import run_reservoir_validation


def load_matlab_data(filepath):
    """
    Loads data from a .mat file and extracts specific variables.

    Args:
        filepath (str): The path to the .mat file.

    Returns:
        tuple: A tuple containing the extracted variables:
               (W_in, res_net, Wout, alpha, kb, n, properties, dt)
               Returns None for a variable if it's not found.
    """
    try:
        data = scipy.io.loadmat(filepath)
        print("Keys in the loaded .mat file:", data.keys())
        print(f"Type of data.get('res_infor'): {type(data.get('res_infor'))}")
        print(f"Content of data.get('res_infor'): {data.get('res_infor')}")
        print(f"Type of data.get('input_infor'): {type(data.get('input_infor'))}")
        print(f"Content of data.get('input_infor'): {data.get('input_infor')}")

        # Extract variables, handling potential KeyError if a variable is missing
        # MATLAB structs are often loaded as numpy structured arrays or dicts.
        # If they are 1x1 arrays, we extract the scalar value.

        W_in_main = None
        res_net_main = None
        alpha_main = None
        kb_main = None
        n_main = None

        res_infor = data.get('res_infor')
        if res_infor is not None:
            # Assuming res_infor is a structured numpy array, typical for structs
            # Access fields like res_infor['field_name'][0,0]
            # If it's a dict, it would be res_infor.get('field_name')
            # Based on typical scipy.io.loadmat behavior for structs, it's often a structured array.
            # Let's try to access elements assuming it's a structured array first.
            # The actual structure might require inspection if this fails.
            try:
                if 'W_in' in res_infor.dtype.names:
                    W_in_main = res_infor['W_in'][0,0]
                if 'res_net' in res_infor.dtype.names:
                    res_net_main = res_infor['res_net'][0,0]
                if 'alpha' in res_infor.dtype.names:
                    alpha_val = res_infor['alpha'][0,0]
                    if isinstance(alpha_val, np.ndarray) and alpha_val.size == 1:
                        alpha_main = alpha_val.item()
                    else:
                        alpha_main = alpha_val
                if 'kb' in res_infor.dtype.names:
                    kb_val = res_infor['kb'][0,0]
                    if isinstance(kb_val, np.ndarray) and kb_val.size == 1:
                        kb_main = kb_val.item()
                    else:
                        kb_main = kb_val
                if 'n' in res_infor.dtype.names:
                    n_val = res_infor['n'][0,0]
                    if isinstance(n_val, np.ndarray) and n_val.size == 1:
                        n_main = n_val.item()
                    else:
                        n_main = n_val
            except Exception as e:
                print(f"Error accessing elements from res_infor (expected structured array): {e}")
                # Fallback or specific handling if it's a plain dict or other structure
                if isinstance(res_infor, dict):
                    W_in_main = res_infor.get('W_in')
                    res_net_main = res_infor.get('res_net')
                    alpha_main = res_infor.get('alpha') # Re-apply item() if needed
                    kb_main = res_infor.get('kb')
                    n_main = res_infor.get('n')


        Wout = data.get('Wout', None)

        properties = data.get('properties', None)
        # Assuming properties is a 1xN or Nx1 array/vector as per MATLAB

        dt = data.get('dt', None)
        if dt is not None and isinstance(dt, np.ndarray) and dt.size == 1:
            dt = dt.item()

        # Parse input_infor from the top-level 'data' dict
        input_infor_parsed = None
        raw_input_infor_from_mat = data.get('input_infor') # Get from the main 'data' dict
        if raw_input_infor_from_mat is not None:
            if isinstance(raw_input_infor_from_mat, np.ndarray) and raw_input_infor_from_mat.dtype == object:
                input_infor_parsed = []
                for item_outer in raw_input_infor_from_mat.flatten(): # Iterate through the object array
                    current_item_to_parse = item_outer
                    # Handle nested arrays if they exist (e.g. for cell arrays of cell arrays of strings)
                    while isinstance(current_item_to_parse, np.ndarray) and current_item_to_parse.size == 1:
                        current_item_to_parse = current_item_to_parse.item(0)

                    if isinstance(current_item_to_parse, str):
                        input_infor_parsed.append(current_item_to_parse)
                if not input_infor_parsed:
                    print("Warning: Parsed input_infor from object array is empty. Defaulting to ['xy', 'qdt'].")
                    input_infor_parsed = ['xy', 'qdt']
            elif isinstance(raw_input_infor_from_mat, list) and all(isinstance(elem, str) for elem in raw_input_infor_from_mat):
                input_infor_parsed = raw_input_infor_from_mat
            elif isinstance(raw_input_infor_from_mat, str):
                 input_infor_parsed = [raw_input_infor_from_mat]
            else:
                print(f"Warning: input_infor type {type(raw_input_infor_from_mat)} not handled by parsing. Defaulting.")
                input_infor_parsed = ['xy', 'qdt']
        else:
            print("Warning: 'input_infor' not found in .mat file. Defaulting to ['xy', 'qdt'].")
            input_infor_parsed = ['xy', 'qdt']

        return W_in_main, res_net_main, Wout, alpha_main, kb_main, n_main, properties, dt, res_infor, input_infor_parsed

    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return (None,) * 10 # Now returning 10 items
    except Exception as e:
        print(f"An error occurred while loading the .mat file: {e}")
        return (None,) * 10 # Now returning 10 items

def matsplit_python(properties_array):
    """
    Unpacks a properties numpy array into individual scalar variables.
    Assumes properties_array is a 1x8 or 8x1 array/vector.

    Args:
        properties_array (np.ndarray): The numpy array loaded from the .mat file.

    Returns:
        tuple: Individual scalar variables (m1, m2, l1, l2, lc1, lc2, I1, I2).
               Returns None for all if input is None or not as expected.
    """
    if properties_array is None:
        print("Error: properties_array is None. Cannot unpack.")
        return (None,) * 8

    # Ensure it's a numpy array
    if not isinstance(properties_array, np.ndarray):
        print("Error: properties_array is not a numpy array.")
        return (None,) * 8

    # Flatten the array to handle both 1xN and Nx1 cases
    properties_flat = properties_array.flatten()

    if properties_flat.shape[0] < 8:
        print(f"Error: properties_array does not contain enough elements (expected 8, got {properties_flat.shape[0]}).")
        return (None,) * 8

    m1 = properties_flat[0]
    m2 = properties_flat[1]
    l1 = properties_flat[2]
    l2 = properties_flat[3]
    lc1 = properties_flat[4]
    lc2 = properties_flat[5]
    I1 = properties_flat[6]
    I2 = properties_flat[7]

    return m1, m2, l1, l2, lc1, lc2, I1, I2

if __name__ == "__main__":
    filepath = './save_file/all_traj_06282022.mat'

    print(f"Attempting to load data from: {filepath}")
    # Corrected unpacking to expect 10 items, including input_infor_loaded
    (W_in_loaded, res_net_loaded, Wout_loaded, alpha_loaded, kb_loaded, n_loaded,
     properties_loaded, dt_loaded, res_infor_loaded, input_infor_loaded) = load_matlab_data(filepath)

    if W_in_loaded is None or res_net_loaded is None or Wout_loaded is None or \
       alpha_loaded is None or kb_loaded is None or n_loaded is None or \
       properties_loaded is None or dt_loaded is None or input_infor_loaded is None:
        print("Essential data not loaded from .mat file. Aborting.")
        # Optionally, exit here if desired:
        # import sys
        # sys.exit("Aborting due to missing essential MAT file data.")
    else:
        print("\n--- Successfully Loaded Essential MAT File Data ---")
        print(f"Type of res_infor_loaded: {type(res_infor_loaded)}")
        print(f"Content of res_infor_loaded: {res_infor_loaded}")
        print(f"Type of input_infor_loaded: {type(input_infor_loaded)}")
        print(f"Content of input_infor_loaded: {input_infor_loaded}")
        print(f"Data dt: {dt_loaded}")
        print(f"Reservoir n: {n_loaded}")
        print(f"Input infor: {input_infor_loaded}")
        # Add other key params if needed

        # --- Parameter Initialization ---
        traj_type = 'infty'  # Test with 'infty'
        bridge_type = 'cubic'

        # Using a smaller val_length for faster testing during development
        # The original MATLAB code might use very large val_length for long simulations.
        sim_val_length = 2000 # adjustable, e.g., 1000, 2000 for tests, up to 200000 for "full" run
        time_infor_sim = {'val_length': sim_val_length, 'dt': dt_loaded}

        failure_params = {'type': 'none', 'amplitude': 0.0, 'amplitude_2': 0.0}
        # disturbance_val = 0.00 # Not directly used yet
        # measurement_noise_val = 0.00 # Not directly used yet

        if traj_type == 'lorenz':
            traj_frequency_sim = 100.0
        elif traj_type == 'circle':
            traj_frequency_sim = 150.0
        elif traj_type == 'infty': # Ensure 'infty' uses 75
            traj_frequency_sim = 75.0
        else: # Default for others
            traj_frequency_sim = 75.0

        print(f"\n--- Simulation Parameters ---")
        print(f"Trajectory Type: {traj_type}")
        print(f"Trajectory Frequency: {traj_frequency_sim} Hz (period factor)")
        print(f"Simulation val_length: {sim_val_length} samples")

        # Initial robot state for trajectory generation (randomized start)
        if traj_type == 'infty':
            q_control_start_sim = np.array([(3-1)*np.random.rand() + 1, (0.0 - 2.4) * np.random.rand()])
        else: # Default for circle, etc.
            q_control_start_sim = np.array([(6-4)*np.random.rand() + 4, (0.0 - 2.4) * np.random.rand() - 0.1])

        qdt_control_start_sim = np.array([0.0, 0.0])
        control_infor_start_sim = {
            'q_control_start': q_control_start_sim,
            'qdt_control_start': qdt_control_start_sim
        }
        print(f"Initial q_start for trajectory generation: {q_control_start_sim}")

        # Split properties for robot_utils and trajectory generation
        m1, m2, l1, l2, lc1, lc2, I1, I2 = matsplit_python(properties_loaded)
        robot_props_for_traj_and_val = {
            'm1': m1, 'm2': m2, 'l1': l1, 'l2': l2,
            'lc1': lc1, 'lc2': lc2, 'I1': I1, 'I2': I2
        }

        # --- Generate Desired Trajectory ---
        print("\n--- Generating Desired Trajectory ---")
        desired_trajectory_data = generate_desired_trajectory(
            traj_type=traj_type,
            bridge_type=bridge_type,
            time_infor=time_infor_sim, # Use sim_val_length
            control_infor_start=control_infor_start_sim,
            robot_properties=robot_props_for_traj_and_val,
            traj_frequency=traj_frequency_sim,
            dt=dt_loaded
        )

        if desired_trajectory_data:
            print("Desired trajectory generated successfully.")
            val_length_actual = desired_trajectory_data['final_val_length']
            print(f"Actual val_length from trajectory generation: {val_length_actual}")

            # --- Run Reservoir Validation ---
            print("\n--- Running Reservoir Validation ---")

            # Determine dim_in for the model
            dim_in_for_model = None
            if res_infor_loaded is not None:
                try:
                    if hasattr(res_infor_loaded, 'dtype') and 'dim_in' in res_infor_loaded.dtype.names:
                        dim_in_for_model = res_infor_loaded['dim_in'][0,0].item()
                    elif isinstance(res_infor_loaded, dict) and 'dim_in' in res_infor_loaded:
                         val = res_infor_loaded['dim_in']
                         if isinstance(val, np.ndarray) and val.size == 1: val = val.item()
                         dim_in_for_model = int(val)
                except Exception as e_resinfo:
                    print(f"Error accessing dim_in from res_infor_loaded: {e_resinfo}")

            if dim_in_for_model is None: # Default if not found in res_infor
                if input_infor_loaded == ['xy', 'qdt']:
                    dim_in_for_model = 8
                elif input_infor_loaded:
                    default_dim_map = {'q':2, 'xy':2, 'qdt':2, 'tau':2}
                    inferred_dim = sum(default_dim_map.get(info_type, 0) for info_type in input_infor_loaded) * 2
                    if inferred_dim > 0: dim_in_for_model = inferred_dim
                    else: dim_in_for_model = 8 # Fallback default
                else: dim_in_for_model = 8 # Fallback if input_infor also missing
            print(f"Derived dim_in_for_model: {dim_in_for_model}")
            print(f"Using dim_in: {dim_in_for_model} for reservoir model.")

            model_params_for_val = {
                'W_in': W_in_loaded, 'res_net': res_net_loaded, 'Wout': Wout_loaded,
                'alpha': alpha_loaded, 'kb': kb_loaded, 'n': n_loaded, 'dt': dt_loaded,
                'dim_in': dim_in_for_model
            }

            desired_data_for_val = {
                'q_control': desired_trajectory_data['q_control'],
                'qdt_control': desired_trajectory_data['qdt_control'],
                'data_control': np.column_stack((desired_trajectory_data['x_control_cartesian'], desired_trajectory_data['y_control_cartesian'])),
                            'final_val_length': val_length_actual # Key changed from 'val_length'
            }

            sim_params_for_val = {
                'start_info_q': desired_trajectory_data['q_control'][0,:],
                'start_info_qdt': desired_trajectory_data['qdt_control'][0,:],
                'start_info_tau': desired_trajectory_data['tau_control'][0,:],
                'r_end_initial': None,
                'failure_type': failure_params['type'],
                # 'disturbance_amplitude': disturbance_val, # Add if used by validation fn
                # 'measurement_noise_amplitude': measurement_noise_val # Add if used
            }

            validation_output = run_reservoir_validation(
                desired_data_for_val, model_params_for_val, robot_props_for_traj_and_val,
                sim_params_for_val, input_infor_loaded
            )

            if validation_output:
                print("Reservoir validation completed. Results summary:")
                for key, value in validation_output.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}")

                # --- Plotting ---
                print("\n--- Plotting Results ---")
                data_control_to_plot = desired_data_for_val['data_control']
                data_pred_to_plot = validation_output['data_pred_cartesian']

                # Ensure val_length_actual is used for slicing, and it's positive
                plot_end_time_idx = val_length_actual # Plot the whole trajectory
                # To plot a segment if too long:
                # plot_end_time_idx = min(val_length_actual, 2000)
                plot_start_time_idx = 0

                if plot_end_time_idx > plot_start_time_idx :
                    plt.figure(figsize=(10, 8))
                    # Plot full trajectories
                    plt.plot(data_control_to_plot[plot_start_time_idx:plot_end_time_idx, 0],
                             data_control_to_plot[plot_start_time_idx:plot_end_time_idx, 1],
                             'r', label='Desired Trajectory', linewidth=1.5)
                    plt.plot(data_pred_to_plot[plot_start_time_idx:plot_end_time_idx, 0],
                             data_pred_to_plot[plot_start_time_idx:plot_end_time_idx, 1],
                             'b--', label='Predicted Trajectory', linewidth=1.0)

                    # Add markers for start points
                    plt.plot(data_control_to_plot[plot_start_time_idx, 0],
                             data_control_to_plot[plot_start_time_idx, 1],
                             'ro', markersize=8, label='Desired Start') # Red circle
                    plt.plot(data_pred_to_plot[plot_start_time_idx, 0],
                             data_pred_to_plot[plot_start_time_idx, 1],
                             'bx', markersize=8, markeredgewidth=2, label='Predicted Start') # Blue X

                    plt.xlabel('x coordinate')
                    plt.ylabel('y coordinate')
                    plt.axhline(0, color='black', linestyle='-.', linewidth=0.5)
                    plt.axvline(0, color='black', linestyle='-.', linewidth=0.5)

                    # Determine plot limits dynamically or use fixed ones
                    # Filter out NaN/Inf values before calculating limits
                    valid_control_x = data_control_to_plot[plot_start_time_idx:plot_end_time_idx, 0]
                    valid_control_y = data_control_to_plot[plot_start_time_idx:plot_end_time_idx, 1]
                    valid_pred_x = data_pred_to_plot[plot_start_time_idx:plot_end_time_idx, 0]
                    valid_pred_y = data_pred_to_plot[plot_start_time_idx:plot_end_time_idx, 1]

                    all_x_finite = np.concatenate((valid_control_x[np.isfinite(valid_control_x)],
                                                   valid_pred_x[np.isfinite(valid_pred_x)]))
                    all_y_finite = np.concatenate((valid_control_y[np.isfinite(valid_control_y)],
                                                   valid_pred_y[np.isfinite(valid_pred_y)]))

                    if all_x_finite.size > 0 and all_y_finite.size > 0:
                        x_center = np.mean(all_x_finite)
                        y_center = np.mean(all_y_finite)

                        x_ptp = np.ptp(all_x_finite)
                        y_ptp = np.ptp(all_y_finite)

                        max_range = max(x_ptp, y_ptp) * 0.6  # Peak-to-peak range * 0.5 + buffer
                        if max_range == 0 or not np.isfinite(max_range): # Handle if all points are same or calculation failed
                            max_range = 1.0

                        plt.xlim([x_center - max_range, x_center + max_range])
                        plt.ylim([y_center - max_range, y_center + max_range])
                    else: # Fallback if no finite data points
                        print("Warning: No finite data to determine plot limits. Using default limits [-1, 1].")
                        plt.xlim([-1, 1])
                        plt.ylim([-1, 1])

                    plt.legend()
                    plt.title(f'Trajectory Tracking: {traj_type}')
                    plt.axis('equal')
                    plt.grid(True)

                    # Save the plot to a file instead of showing
                    plot_filename = f"./{traj_type}_trajectory_plot.png"
                    plt.savefig(plot_filename)
                    print(f"Plot saved to {plot_filename}")
                    plt.close() # Close the figure to free memory
                else:
                    print("Not enough data points to plot.")
            else:
                print("Reservoir validation did not return results.")
        else:
            print("Desired trajectory generation failed. Skipping validation and plotting.")

    # Final check for loaded data for completeness
    if dt_loaded is None: print("Warning: dt_loaded is None.")

    print("\n--- Main script execution finished ---")
    print("it might be due to the .mat file not existing at the specified path,")
    print("or the variable names/structure within the .mat file differing from expectations.")
    print("The 'Keys in the loaded .mat file' printout above should help identify actual variable names.")
