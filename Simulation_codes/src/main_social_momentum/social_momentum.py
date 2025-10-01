# social_momentum.py
"""
Module implementing the core Social Momentum algorithm for robot action selection.
"""
import numpy as np
from typing import List, Tuple, Optional

# Use geometry utils for calculations
import geometry_utils as geo

# --- Algorithm Constants ---
DEFAULT_FOV_DEG = 180         # Field of view for reactive agents (degrees)

def filter_colliding_actions(
    robot_q: np.ndarray,
    robot_actions: List[np.ndarray],
    humans_q: List[np.ndarray],
    humans_v: List[np.ndarray],
    time_step: float,
    robot_radius: float,
    human_radius: float
) -> List[np.ndarray]:
    """
    Filters the robot's action space, removing actions that lead to collision.
    Uses check_collision from geometry_utils.

    Args:
        robot_q: Current robot position.
        robot_actions: List of possible robot velocity actions.
        humans_q: List of current human positions.
        humans_v: List of current human velocities.
        time_step: Prediction time step.
        robot_radius: Robot collision radius.
        human_radius: Human collision radius.

    Returns:
        A list of collision-free robot actions.
    """
    v_cf = []
    for action in robot_actions:
        collision_predicted = False
        for hq, hv in zip(humans_q, humans_v):
            if geo.check_collision(robot_q, action, hq, hv, time_step, robot_radius, human_radius):
                collision_predicted = True
                break # No need to check other humans for this action
        if not collision_predicted:
            v_cf.append(action)
    return v_cf


def update_reactive_agents(
    robot_q: np.ndarray,
    current_robot_velocity: np.ndarray,
    humans_q: List[np.ndarray],
    humans_v: List[np.ndarray],
    fov_deg: float = DEFAULT_FOV_DEG
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Identifies human agents within the robot's field of view based on its direction of motion.

    Args:
        robot_q: Current robot position.
        current_robot_velocity: Robot's current velocity vector (determines forward direction).
        humans_q: List of all human positions.
        humans_v: List of all human velocities.
        fov_deg: Robot's field of view in degrees.

    Returns:
        A tuple containing:
        - List of reactive human positions.
        - List of reactive human velocities.
        - List of indices of reactive humans in the original lists.
    """
    reactive_q = []
    reactive_v = []
    reactive_indices = []

    robot_dir, robot_speed = geo.normalize(current_robot_velocity)

    if robot_speed < geo.EPSILON:
        return reactive_q, reactive_v, reactive_indices # Return empty lists if robot not moving

    fov_rad_half = np.deg2rad(fov_deg) / 2.0

    for i, (hq, hv) in enumerate(zip(humans_q, humans_v)):
        vec_rh = np.asarray(hq) - np.asarray(robot_q)
        vec_rh_normalized, dist_rh = geo.normalize(vec_rh)

        if dist_rh < geo.EPSILON:
            continue # Skip human at the same position

        # Angle between robot direction and vector to human
        dot_product = np.dot(robot_dir, vec_rh_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0) # Ensure valid input for arccos
        angle = np.arccos(dot_product)

        if abs(angle) <= fov_rad_half:
            reactive_q.append(hq)
            reactive_v.append(hv)
            reactive_indices.append(i)

    return reactive_q, reactive_v, reactive_indices

def calculate_efficiency_score(action: np.ndarray, robot_q: np.ndarray, goal_q: np.ndarray, time_step: float) -> float:
    """
    Calculates the efficiency score for an action (progress towards goal).
    Higher score is better. Uses negative distance to goal after one step.

    Args:
        action: The robot action (velocity) being considered.
        robot_q: Current robot position.
        goal_q: Robot's goal position.
        time_step: Simulation time step.

    Returns:
        Efficiency score.
    """
    robot_q_next = np.asarray(robot_q) + np.asarray(action) * time_step
    dist_to_goal = np.linalg.norm(robot_q_next - np.asarray(goal_q))
    # Negative distance: smaller distance -> larger (less negative) score
    return -dist_to_goal


def calculate_social_momentum_score(
    robot_action: np.ndarray,
    robot_q: np.ndarray,
    current_robot_velocity: np.ndarray,
    reactive_humans_q: List[np.ndarray],
    reactive_humans_v: List[np.ndarray],
    time_step: float
) -> float:
    """
    Calculates the Social Momentum objective L(vr) from Eq. 4 & surrounding text.
    Uses calculate_angular_momentum_z from geometry_utils.

    Args:
        robot_action: The robot action (velocity) being evaluated.
        robot_q: Current robot position.
        current_robot_velocity: Robot's current velocity.
        reactive_humans_q: List of reactive human positions.
        reactive_humans_v: List of reactive human velocities.
        time_step: Prediction time step.

    Returns:
        The social momentum score for the given action. Returns 0 if momentum sign flips.
    """
    total_sm_score = 0.0
    robot_q = np.asarray(robot_q)
    robot_action = np.asarray(robot_action)
    current_robot_velocity = np.asarray(current_robot_velocity)
    robot_q_next = robot_q + robot_action * time_step

    weights = []
    projected_momenta_z = []

    # First pass: Calculate weights and check for sign flips
    for i, (hq, hv) in enumerate(zip(reactive_humans_q, reactive_humans_v)):
        hq = np.asarray(hq)
        hv = np.asarray(hv)
        hq_next = hq + hv * time_step

        # Current momentum L_rhi
        L_current_z = geo.calculate_angular_momentum_z(robot_q, current_robot_velocity, hq, hv)

        # Projected momentum L^_rhi(vr)
        L_projected_z = geo.calculate_angular_momentum_z(robot_q_next, robot_action, hq_next, hv)

        # Check sign consistency (Eq. 4 condition)
        if L_current_z * L_projected_z < -geo.EPSILON: # Use -EPSILON to avoid float issues near zero
            return 0.0 # Sign flipped -> score is 0

        dist = np.linalg.norm(robot_q - hq)
        weight = 1.0 / (dist + geo.EPSILON) # Weight: inverse distance
        weights.append(weight)
        projected_momenta_z.append(L_projected_z)

    total_weight = sum(weights)
    if total_weight < geo.EPSILON:
        return 0.0 # No agents close enough or weighted to contribute

    normalized_weights = [w / total_weight for w in weights]

    # Second pass: Calculate weighted score
    for i in range(len(reactive_humans_q)):
         total_sm_score += normalized_weights[i] * abs(projected_momenta_z[i])

    return total_sm_score


def select_social_momentum_action(
    robot_q: np.ndarray,
    current_robot_velocity: np.ndarray,
    robot_goal_q: np.ndarray,
    all_humans_q: List[np.ndarray],
    all_humans_v: List[np.ndarray],
    robot_action_space: List[np.ndarray],
    lambda_sm: float, # Weight for social momentum term (λ in Eq. 5)
    time_step: float,
    robot_radius: float,
    human_radius: float,
    fov_deg: float = DEFAULT_FOV_DEG
) -> np.ndarray:
    """
    Selects the best robot action based on the Social Momentum framework (Algorithm 1).

    Args:
        robot_q: Current robot position [x, y].
        current_robot_velocity: Robot's current velocity [vx, vy].
        robot_goal_q: Robot's destination position [x, y].
        all_humans_q: List of all current human positions [[x, y], ...].
        all_humans_v: List of all current human velocities [[vx, vy], ...].
        robot_action_space: List of possible discrete robot velocity actions [[vx, vy], ...].
        lambda_sm: Weighting factor for the social momentum objective.
        time_step: Simulation/prediction time step.
        robot_radius: Collision radius of the robot.
        human_radius: Collision radius of humans.
        fov_deg: Robot's field of view in degrees for reactivity.

    Returns:
        The selected optimal robot action (velocity vector [vx, vy]),
        or np.array([0.0, 0.0]) if no valid (collision-free) action is found.
    """
    # 1. Filter action space for collisions (Vcf)
    v_cf = filter_colliding_actions(
        robot_q, robot_action_space, all_humans_q, all_humans_v,
        time_step, robot_radius, human_radius
    )

    if not v_cf:
        return np.array([0.0, 0.0]) # Stop if no safe move

    # 2. Update reactive agents (Find R)
    reactive_q, reactive_v, _ = update_reactive_agents(
        robot_q, current_robot_velocity, all_humans_q, all_humans_v, fov_deg
    )

    best_action = None
    best_score = -np.inf

    # 3. Optimize based on whether reactive agents exist
    if reactive_q:
        # Optimize Momentum (Combine Efficiency E and Social Momentum L)
        for action in v_cf:
            efficiency_score = calculate_efficiency_score(action, robot_q, robot_goal_q, time_step)
            sm_score = calculate_social_momentum_score(
                action, robot_q, current_robot_velocity, reactive_q, reactive_v, time_step
            )
            total_score = efficiency_score + lambda_sm * sm_score

            if total_score > best_score:
                best_score = total_score
                best_action = action
    else:
        # Optimize Efficiency only (No reactive agents in FOV)
        for action in v_cf:
            efficiency_score = calculate_efficiency_score(action, robot_q, robot_goal_q, time_step)
            if efficiency_score > best_score:
                best_score = efficiency_score
                best_action = action

    # Fallback if scoring failed or all scores were -inf
    if best_action is None and v_cf:
        best_action = v_cf[0] # Default to first safe action
    elif best_action is None: # Should only happen if v_cf was empty initially
         best_action = np.array([0.0, 0.0])

    return np.asarray(best_action)
