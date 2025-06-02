import numpy as np

def forward_dynamics(q, qdt, tau, m1, m2, l1, l2, lc1, lc2, I1, I2):
    """
    Calculates the forward dynamics of a 2-DOF robot manipulator.

    Args:
        q (np.ndarray): Current joint angles (2x1 array: [q1, q2]).
        qdt (np.ndarray): Current joint velocities (2x1 array: [qdt1, qdt2]).
        tau (np.ndarray): Motor torques (2x1 array: [tau1, tau2]).
        m1, m2: Masses of the links.
        l1, l2: Lengths of the links.
        lc1, lc2: Distances to the center of mass of the links.
        I1, I2: Moments of inertia of the links.

    Returns:
        np.ndarray: Joint accelerations (q2dt, a 2x1 array: [q2dt1, q2dt2]).
    """
    q1 = q[0,0]
    q2 = q[1,0]
    qdt1 = qdt[0,0]
    qdt2 = qdt[1,0]
    tau1 = tau[0,0]
    tau2 = tau[1,0]

    # H-matrix components (Inertia matrix)
    H11 = m1 * lc1**2 + I1 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(q2)) + I2
    H12 = m2 * l1 * lc2 * np.cos(q2) + m2 * lc2**2 + I2
    H21 = H12
    H22 = m2 * lc2**2 + I2

    # h-term factor (used in Coriolis and centrifugal force calculations)
    h_term_factor = m2 * l1 * lc2 * np.sin(q2)

    C_qd_1 = -h_term_factor * qdt2 * qdt1 - h_term_factor * (qdt1 + qdt2) * qdt2
    C_qd_2 =  h_term_factor * qdt1**2

    rhs = np.array([
        [tau1 - C_qd_1],
        [tau2 - C_qd_2]
    ])

    H = np.array([[H11, H12], [H21, H22]])

    try:
        q2dt = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix H in forward_dynamics. Returning zero acceleration.")
        q2dt = np.zeros((2,1))

    return q2dt

def inverse_kinematics(x, y, l1, l2):
    """
    Calculates inverse kinematics for a 2-DOF planar robot.
    Args:
        x (float): Desired x-coordinate of the end-effector.
        y (float): Desired y-coordinate of the end-effector.
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.
    Returns:
        tuple: (q1_ik, q2_ik) joint angles in radians.
               Returns (None, None) if position is unreachable.
    """
    cos_q2_numerator = x**2 + y**2 - l1**2 - l2**2
    cos_q2_denominator = 2 * l1 * l2

    if cos_q2_denominator == 0:
        return None, None

    cos_q2_val = cos_q2_numerator / cos_q2_denominator

    if not (-1 <= cos_q2_val <= 1):
        return None, None

    q2_ik = np.arccos(cos_q2_val)

    numerator_q1_second_term = l2 * np.sin(q2_ik)
    denominator_q1_second_term = l1 + l2 * np.cos(q2_ik)

    term1_q1 = np.arctan2(y, x)
    term2_q1 = np.arctan2(numerator_q1_second_term, denominator_q1_second_term)

    q1_ik = term1_q1 - term2_q1

    # The MATLAB code's q1 adjustment block is not directly translated here
    # as np.arctan2 typically handles quadrants correctly for the principal solution.
    # If a specific range like [0, 2*pi] or a different solution branch is needed,
    # further adjustments might be required by the caller or by refining this.
    return q1_ik, q2_ik

def forward_kinematics(q1, q2, l1, l2):
    """
    Calculates forward kinematics for a 2-DOF planar robot.
    Args:
        q1 (float): Joint angle of the first link.
        q2 (float): Joint angle of the second link.
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.
    Returns:
        tuple: (x, y) Cartesian coordinates of the end-effector.
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return x, y

def generate_cubic_bridge_trajectory(q_start, qdt_start,
                                     x_target_point, y_target_point,
                                     x_target_next_point, y_target_next_point,
                                     l1, l2, dt, bridge_duration):
    """
    Generates a cubic polynomial bridge trajectory in joint space and converts it to Cartesian space.

    Args:
        q_start (np.ndarray): Initial joint angles [q1_start, q2_start]. (1D array of 2 elements)
        qdt_start (np.ndarray): Initial joint velocities [q1dt_start, q2dt_start]. (1D array of 2 elements)
        x_target_point, y_target_point (float): Cartesian coordinates of the target point on the main trajectory.
        x_target_next_point, y_target_next_point (float): Cartesian coordinates of the next point on the main trajectory.
        l1, l2 (float): Robot link lengths.
        dt (float): Timestep.
        bridge_duration (float): Total time for the bridge trajectory.

    Returns:
        tuple: (q_bridge_trajectory, xy_bridge_trajectory)
               q_bridge_trajectory (np.ndarray): Joint space trajectory (samples x 2).
               xy_bridge_trajectory (np.ndarray): Cartesian space trajectory (samples x 2).
               Returns (None, None) if IK fails.
    """
    q_start = np.asarray(q_start)
    qdt_start = np.asarray(qdt_start)

    # 1. Calculate target joint configuration (q_target, qdt_target)
    # Perform inverse kinematics for (x_target_point, y_target_point)
    q1_target_raw, q2_target_raw = inverse_kinematics(x_target_point, y_target_point, l1, l2)
    if q1_target_raw is None:
        print("IK failed for target point in bridge generation.")
        return None, None

    # Adjust q2 sign based on q_start[1] (theta0(2) in MATLAB)
    if q_start[1] < 0:
        if q2_target_raw > 0: # Ensure we pick the negative solution if q_start[1] is negative
             # This implies we need the other solution from acos.
             # If acos((X)/(Y)) = V, then the other solution is -V (within 2pi range)
             # Or, it might mean recalculating q1 if q2 sign flips.
             # The MATLAB code: q2_bg(1)=acos(...); if theta0(2)<0 q2_bg = -q2_bg; end; q1_bg = atan(...) - atan(...)
             # This suggests q2's sign is flipped, then q1 is recalculated.
             # For simplicity, let's re-calculate IK assuming the other q2 solution.
             # The primary solution from np.arccos is [0, pi]. If q_start[1] < 0, we want q2_target in [-pi, 0).
             # So, if q2_target_raw is positive, make it negative.
             q2_target = -q2_target_raw
             # Recalculate q1 with the new q2_target
             # term1_q1 = np.arctan2(y_target_point, x_target_point)
             # term2_q1_num = l2 * np.sin(q2_target)
             # term2_q1_den = l1 + l2 * np.cos(q2_target)
             # q1_target = term1_q1 - np.arctan2(term2_q1_num, term2_q1_den)
             # Simpler: just use the negative of raw q2 and re-evaluate q1 using the original IK logic with this choice.
             # However, our current IK only returns one solution.
             # A robust IK would return both, or take a preference.
             # For now, let's stick to the MATLAB's direct approach:
             # q2_target = np.arccos((x_target_point**2 + y_target_point**2 - l1**2 - l2**2) / (2 * l1 * l2))
             # if q_start[1] < 0 and q2_target > 0: # Check if q2_target was positive from acos
             #    q2_target = -q2_target
             # q1_target = np.arctan2(y_target_point, x_target_point) - \
             #             np.arctan2(l2 * np.sin(q2_target), l1 + l2 * np.cos(q2_target))
             # This is what inverse_kinematics already does for the positive q2.
             # If q_start[1] < 0, and q2_target_raw (which is from [0,pi]) is positive, we use -q2_target_raw.
             # Then we must recalculate q1 based on this new q2.
             # Our IK function structure needs to support choosing the q2 solution or returning both.
             # Let's assume for now the IK function gives a q2 in [0,pi].
             # If q_start[1] < 0, we simply take the negative of the q2 from IK, and recompute q1.
             # This matches the MATLAB snippet logic.

             # Simplified approach for now:
             # If q_start[1] is negative, and the returned q2_target_raw is positive, we assume the other solution for q2.
             # This is a common convention.
             if q2_target_raw > 1e-6: # Check if it's meaningfully positive
                 q2_target = -q2_target_raw
                 # Recalculate q1 based on this new q2_target
                 # x = l1*cos(q1) + l2*cos(q1+q2) -> l1*cos(q1) = x - l2*cos(q1+q2)
                 # y = l1*sin(q1) + l2*sin(q1+q2) -> l1*sin(q1) = y - l2*sin(q1+q2)
                 # atan2( (y - l2*sin(q1+q2)) / l1, (x - l2*cos(q1+q2)) / l1 ) but this is q1+q2...
                 # The standard q1 formula used in inverse_kinematics should be re-evaluated:
                 q1_target = np.arctan2(y_target_point, x_target_point) - \
                             np.arctan2(l2 * np.sin(q2_target), l1 + l2 * np.cos(q2_target))
             else: # q2_target_raw is zero or negative (should be only zero from acos)
                 q1_target = q1_target_raw
                 q2_target = q2_target_raw
        else: # q_start[1] is positive or zero, use the IK solution as is (q2_target_raw is in [0, pi])
            q1_target = q1_target_raw
            q2_target = q2_target_raw
    else: # q_start[1] >= 0
        q1_target = q1_target_raw
        q2_target = q2_target_raw

    q_target = np.array([q1_target, q2_target])

    # Perform inverse kinematics for (x_target_next_point, y_target_next_point)
    q1_target_next_raw, q2_target_next_raw = inverse_kinematics(x_target_next_point, y_target_next_point, l1, l2)
    if q1_target_next_raw is None:
        print("IK failed for target_next point in bridge generation.")
        return None, None

    # Adjust q2_target_next sign based on q_target[1] (similar logic to above)
    if q_target[1] < 0:
        if q2_target_next_raw > 1e-6:
            q2_target_next = -q2_target_next_raw
            q1_target_next = np.arctan2(y_target_next_point, x_target_next_point) - \
                               np.arctan2(l2 * np.sin(q2_target_next), l1 + l2 * np.cos(q2_target_next))
        else:
            q1_target_next = q1_target_next_raw
            q2_target_next = q2_target_next_raw
    else:
        q1_target_next = q1_target_next_raw
        q2_target_next = q2_target_next_raw

    q_target_next = np.array([q1_target_next, q2_target_next])

    # Estimate target joint velocities
    qdt_target = (q_target_next - q_target) / dt

    # 2. Calculate cubic polynomial coefficients (a0, a1, a2, a3) for each joint
    # T = bridge_duration
    # a0 = q_start
    # a1 = qdt_start
    # a2 = (3*(q_target - q_start)/T**2) - (2*qdt_start/T) - (qdt_target/T)
    # a3 = (-2*(q_target - q_start)/T**3) + ((qdt_target + qdt_start)/T**2)

    T = bridge_duration
    if T == 0: # Avoid division by zero if duration is zero
        # If duration is zero, output might be just the start or end point.
        # For now, let's assume duration > 0. Or handle as a special case.
        # Let's return just the start point repeated twice if T=0 for q, and corresponding FK for xy
        print("Warning: Bridge duration is zero. Returning start point.")
        q_bridge_trajectory = np.array([q_start, q_start])
        x_start_fk, y_start_fk = forward_kinematics(q_start[0], q_start[1], l1, l2)
        xy_bridge_trajectory = np.array([[x_start_fk, y_start_fk], [x_start_fk, y_start_fk]])
        return q_bridge_trajectory, xy_bridge_trajectory

    a0 = q_start
    a1 = qdt_start

    # Element-wise operations for a2 and a3
    T_sq = T**2
    T_cub = T**3

    a2 = (3 * (q_target - q_start) / T_sq) - (2 * qdt_start / T) - (qdt_target / T)
    a3 = (-2 * (q_target - q_start) / T_cub) + ((qdt_target + qdt_start) / T_sq)

    # 3. Generate the bridge trajectory points in joint space
    # t_bg = 0:dt:bridge_time in MATLAB
    # Using np.arange, the endpoint T might be excluded if T is a multiple of dt.
    # np.linspace is often better for including endpoints.
    # num_points = int(round(T / dt)) + 1
    # t_bg = np.linspace(0, T, num_points, endpoint=True)
    # The MATLAB 0:dt:T includes T. np.arange(0, T + dt, dt) is a common way.
    t_bg = np.arange(0, T + dt/2, dt) # dt/2 to ensure T is included if it's a multiple of dt
    if t_bg[-1] > T: # Ensure we don't overshoot T by too much due to floating point
        t_bg = t_bg[t_bg <= T]
        if t_bg[-1] < T - dt/2 and T > 0 : # if T was not included and should have been
             t_bg = np.append(t_bg, T)


    q_bridge_trajectory = np.zeros((len(t_bg), 2))
    for i in range(2): # For each joint
        q_bridge_trajectory[:, i] = a0[i] + a1[i]*t_bg + a2[i]*(t_bg**2) + a3[i]*(t_bg**3)

    # 4. Convert bridge joint trajectory to Cartesian space
    xy_bridge_trajectory = np.zeros((len(t_bg), 2))
    for i in range(len(t_bg)):
        q1_curr, q2_curr = q_bridge_trajectory[i, 0], q_bridge_trajectory[i, 1]
        x_curr, y_curr = forward_kinematics(q1_curr, q2_curr, l1, l2)
        xy_bridge_trajectory[i, :] = [x_curr, y_curr]

    return q_bridge_trajectory, xy_bridge_trajectory

def normalize_python(data, target_range):
    """
    Normalizes data to a target range. Mimics MATLAB's normalize(data, 'range', target_range).
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input 'data' must be a numpy array.")
    if len(target_range) != 2:
        raise ValueError("'target_range' must have two elements.")

    data_min = np.min(data)
    data_max = np.max(data)

    target_min = target_range[0]
    target_max = target_range[1]

    data_range = data_max - data_min

    if data_range == 0:
        return np.full(data.shape, target_min)

    normalized_data = (data - data_min) / data_range
    normalized_data = normalized_data * (target_max - target_min) + target_min

    return normalized_data
