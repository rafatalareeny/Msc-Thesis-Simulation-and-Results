import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random # Needed for random initial states

# --- Constants and Configuration ---
# These should be tuned for your specific robot and scenario
DEFAULT_TIME_STEP = 0.1       # Simulation/Prediction time step (s) - Smaller for smoother animation
DEFAULT_ROBOT_RADIUS = 0.3   # meters
DEFAULT_HUMAN_RADIUS = 0.35   # meters
DEFAULT_FOV_DEG = 180         # Field of view for reactive agents (degrees)
EPSILON = 1e-6                # Small number to avoid division by zero

# --- Helper Functions ---

def calculate_angular_momentum_z(q1: np.ndarray, v1: np.ndarray,
                                 q2: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the Z-component of the angular momentum for a two-agent system.
    Assumes unit mass for both agents.

    Args:
        q1: Position vector [x, y] of agent 1.
        v1: Velocity vector [x, y] of agent 1.
        q2: Position vector [x, y] of agent 2.
        v2: Velocity vector [x, y] of agent 2.

    Returns:
        The Z-component of the total angular momentum.
    """
    if q1 is None or v1 is None or q2 is None or v2 is None:
        return 0.0
    if np.array_equal(q1, q2): # Avoid issues if agents are at the same spot
        return 0.0

    pc = (q1 + q2) / 2.0  # Center of mass
    p1c = q1 - pc        # Position relative to center of mass
    p2c = q2 - pc

    # 2D cross product: Lz = x*vy - y*vx
    L1_z = p1c[0] * v1[1] - p1c[1] * v1[0]
    L2_z = p2c[0] * v2[1] - p2c[1] * v2[0]

    return L1_z + L2_z

def check_collision(robot_q: np.ndarray, robot_action: np.ndarray,
                    human_q: np.ndarray, human_v: np.ndarray,
                    time_step: float, robot_radius: float, human_radius: float) -> bool:
    """
    Checks for collision between robot and one human for a given robot action.

    Args:
        robot_q: Current robot position [x, y].
        robot_action: Proposed robot velocity [vx, vy].
        human_q: Current human position [x, y].
        human_v: Current human velocity [vx, vy].
        time_step: Prediction time step.
        robot_radius: Robot collision radius.
        human_radius: Human collision radius.

    Returns:
        True if a collision is predicted, False otherwise.
    """
    # Simplified check: predict positions at t + time_step
    robot_q_next = robot_q + robot_action * time_step
    human_q_next = human_q + human_v * time_step

    min_dist_sq = (robot_radius + human_radius)**2

    dist_sq = np.sum((robot_q_next - human_q_next)**2)

    return dist_sq < min_dist_sq

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

    Args:
        robot_q: Current robot position.
        robot_actions: List of possible robot velocity actions.
        humans_q: List of current human positions.
        humans_v: List of current human velocities.
        time_step: Prediction time step.
        robot_radius: Robot collision radius.
        human_radius: Human collision radius.

    Returns:
        A list of collision-free robot actions (Vcf from Algorithm 1).
    """
    v_cf = []
    for action in robot_actions:
        collision_predicted = False
        for hq, hv in zip(humans_q, humans_v):
            if check_collision(robot_q, action, hq, hv, time_step, robot_radius, human_radius):
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

    robot_speed = np.linalg.norm(current_robot_velocity)
    if robot_speed < EPSILON:
        # If robot is stationary, consider all agents potentially reactive
        # Or define a default forward direction (e.g., towards goal)
        # For simplicity here, we'll consider none reactive if not moving
        # print("Warning: Robot velocity is near zero. Cannot determine FOV reliably.")
        return reactive_q, reactive_v, reactive_indices # Return empty lists

    robot_dir = current_robot_velocity / robot_speed
    fov_rad_half = np.deg2rad(fov_deg) / 2.0

    for i, (hq, hv) in enumerate(zip(humans_q, humans_v)):
        vec_rh = hq - robot_q
        dist_rh = np.linalg.norm(vec_rh)
        if dist_rh < EPSILON:
            continue # Skip human at the same position

        vec_rh_normalized = vec_rh / dist_rh

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
    robot_q_next = robot_q + action * time_step
    dist_to_goal = np.linalg.norm(robot_q_next - goal_q)
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

    Args:
        robot_action: The robot action (velocity) being evaluated.
        robot_q: Current robot position.
        current_robot_velocity: Robot's current velocity.
        reactive_humans_q: List of reactive human positions.
        reactive_humans_v: List of reactive human velocities.
        time_step: Prediction time step.

    Returns:
        The social momentum score for the given action.
    """
    total_sm_score = 0.0
    robot_q_next = robot_q + robot_action * time_step

    weights = []
    projected_momenta_z = []

    # First pass: Calculate weights and check for sign flips
    for i, (hq, hv) in enumerate(zip(reactive_humans_q, reactive_humans_v)):
        hq_next = hq + hv * time_step

        # Current momentum L_rhi
        L_current_z = calculate_angular_momentum_z(robot_q, current_robot_velocity, hq, hv)

        # Projected momentum L^_rhi(vr)
        L_projected_z = calculate_angular_momentum_z(robot_q_next, robot_action, hq_next, hv)

        # Check sign consistency (Eq. 4 condition)
        # L_current * L_projected >= 0 means sign is preserved or one is zero
        if L_current_z * L_projected_z < -EPSILON: # Use -EPSILON to avoid float issues near zero
             # Sign flipped for at least one agent -> score is 0
            return 0.0

        dist = np.linalg.norm(robot_q - hq)
        # Weight: inverse distance (closer agents are more important)
        weight = 1.0 / (dist + EPSILON)
        weights.append(weight)
        projected_momenta_z.append(L_projected_z)

    # Normalize weights if needed (paper mentions normalization in sec 4.3.3, though not explicitly in Eq 4)
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return 0.0 # No agents close enough to contribute weight

    normalized_weights = [w / total_weight for w in weights]

    # Second pass: Calculate weighted score
    for i in range(len(reactive_humans_q)):
         total_sm_score += normalized_weights[i] * abs(projected_momenta_z[i])

    return total_sm_score


# --- Main Algorithm Function ---

def select_social_momentum_action(
    robot_q: np.ndarray,
    current_robot_velocity: np.ndarray,
    robot_goal_q: np.ndarray,
    all_humans_q: List[np.ndarray],
    all_humans_v: List[np.ndarray],
    robot_action_space: List[np.ndarray],
    lambda_sm: float, # Weight for social momentum term (λ in Eq. 5)
    time_step: float = DEFAULT_TIME_STEP,
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    human_radius: float = DEFAULT_HUMAN_RADIUS,
    fov_deg: float = DEFAULT_FOV_DEG
) -> Optional[np.ndarray]:
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
        # print("Warning: No collision-free actions found! Stopping.")
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
        # print(f"Optimizing with {len(reactive_q)} reactive agents.")
        for action in v_cf:
            efficiency_score = calculate_efficiency_score(action, robot_q, robot_goal_q, time_step)
            sm_score = calculate_social_momentum_score(
                action, robot_q, current_robot_velocity, reactive_q, reactive_v, time_step
            )
            # Combined score (Eq. 5)
            total_score = efficiency_score + lambda_sm * sm_score

            if total_score > best_score:
                best_score = total_score
                best_action = action
    else:
        # Optimize Efficiency only (No reactive agents in FOV)
        # print("Optimizing for efficiency only (no reactive agents).")
        for action in v_cf:
            efficiency_score = calculate_efficiency_score(action, robot_q, robot_goal_q, time_step)
            if efficiency_score > best_score:
                best_score = efficiency_score
                best_action = action

    # Ensure an action is selected if v_cf was not empty
    if best_action is None and v_cf:
        # Default to the first collision-free action if scoring failed somehow
        # or if all scores were -inf (e.g., goal is unreachable)
        # A better fallback might be the action closest to current velocity or goal
        # print("Warning: Scoring resulted in no best action, picking first safe one.")
        best_action = v_cf[0]
    elif best_action is None: # Should only happen if v_cf was empty initially
         best_action = np.array([0.0, 0.0])


    return best_action


# --- Simulation and Visualization ---
if __name__ == "__main__":

    # --- Simulation Parameters ---
    SIM_TIME_STEP = DEFAULT_TIME_STEP # Use consistent time step
    TOTAL_SIM_TIME = 30.0 # seconds
    HALLWAY_WIDTH = 1.5   # meters
    PLOT_LENGTH = 15.0    # meters (for y-axis limits)
    ROBOT_MAX_SPEED = 2.2 # meters/second
    HUMAN_MAX_SPEED = 2.0 # meters/second
    SOCIAL_MOMENTUM_WEIGHT = 0.015 # Adjust this weight as needed

    # --- Initialize Robot State ---
    center_x = 0.0
    robot_start_x = 0.1 #random.uniform(-HALLWAY_WIDTH/2 * 0.8, HALLWAY_WIDTH/2 * 0.8)
    robot_pos = np.array([robot_start_x, 0.0])
    robot_goal = np.array([center_x, PLOT_LENGTH])
    # robot_pos = np.array([robot_start_x, 0.0]) # Start near y=0
    # robot_goal = np.array([robot_start_x, PLOT_LENGTH * 0.9]) # Goal near the end of the plot

    # Random initial velocity for robot
    robot_vel = np.array([0.0, ROBOT_MAX_SPEED])
    # start_speed_r = random.uniform(0.5, ROBOT_MAX_SPEED)
    # start_angle_r = random.uniform(0, 2 * np.pi) # Random direction
    # robot_vel = np.array([start_speed_r * np.cos(start_angle_r), start_speed_r 
    # * np.sin(start_angle_r)])
    # # Ensure initial velocity isn't excessively sideways for a hallway scenario
    # robot_vel[0] *= 0.3 # Reduce initial side-to-side speed
    # robot_vel = np.clip(robot_vel, -ROBOT_MAX_SPEED, ROBOT_MAX_SPEED) # Ensure max speed limit


    # --- Initialize Human State (One Human) ---
    # human_pos = np.array([center_x, PLOT_LENGTH / 2])
    human_start_x = random.uniform(-HALLWAY_WIDTH/2 * 0.8, HALLWAY_WIDTH/2 * 0.8)
    # Place human further down the hallway initially
    human_start_y = random.uniform(PLOT_LENGTH * 0.3, PLOT_LENGTH * 0.7)
    human_pos = np.array([human_start_x, human_start_y])

    # Random initial velocity for human
    # human_vel = np.array([0.0, -HUMAN_MAX_SPEED])
    start_speed_h = random.uniform(0.5, HUMAN_MAX_SPEED)
    start_angle_h = random.uniform(0, 2 * np.pi) # Random direction
    human_vel = np.array([start_speed_h * np.cos(start_angle_h), start_speed_h * np.sin(start_angle_h)])
     # Ensure initial velocity isn't excessively sideways
    human_vel[0] *= 0.3
    human_vel = np.clip(human_vel, -HUMAN_MAX_SPEED, HUMAN_MAX_SPEED)

    # Make lists for the algorithm functions
    human_positions = [human_pos]
    human_velocities = [human_vel]

    # --- Robot's possible actions ---
    # More granular action space might be better
    angles = np.linspace(-np.pi / 4, np.pi / 4, 5) # Turn angles from -45 to +45 deg
    speeds = [ROBOT_MAX_SPEED * 0.5, ROBOT_MAX_SPEED]
    possible_actions = [np.array([0.0, 0.0])] # Include stopping
    for speed in speeds:
        for angle in angles:
            action = np.array([speed * np.sin(angle), speed * np.cos(angle)]) # Assume forward is +y
            possible_actions.append(action)

    # --- Setup Plot ---
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(-HALLWAY_WIDTH / 2 - 0.5, HALLWAY_WIDTH / 2 + 0.5)
    ax.set_ylim(-1, PLOT_LENGTH)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Classification of Human Obstruction Corridor Width = 1m')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)

    # Draw Hallway Lines
    ax.axvline(-HALLWAY_WIDTH / 2, color='k', linestyle='--', label='Hallway Wall')
    ax.axvline(HALLWAY_WIDTH / 2, color='k', linestyle='--')

    # Plot initial positions and goal
    robot_dot, = ax.plot([], [], 'ro', markersize=8, label='Robot') # Red dot
    human_dot, = ax.plot([], [], 'bo', markersize=8, label='Human') # Blue dot
    goal_marker, = ax.plot(robot_goal[0], robot_goal[1], 'gx', markersize=10, label='Goal')

    # Time display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.legend(loc='upper right')

    # Store history for trails (optional)
    robot_traj_x, robot_traj_y = [], []
    human_traj_x, human_traj_y = [], []
    trail_line_r, = ax.plot([], [], 'r:', alpha=0.5) # Robot trail
    trail_line_h, = ax.plot([], [], 'b:', alpha=0.5) # Human trail

    # --- Animation Update Function ---
    current_time = 0.0

    def update(frame):
        global robot_pos, robot_vel, human_pos, human_vel, current_time
        global robot_traj_x, robot_traj_y, human_traj_x, human_traj_y # To modify history

        if current_time > TOTAL_SIM_TIME:
             print("Simulation time limit reached.")
             ani.event_source.stop() # Stop the animation
             return robot_dot, human_dot, time_text, trail_line_r, trail_line_h

        # Add robot's current velocity to potential actions if moving
        current_vel_norm = np.linalg.norm(robot_vel)
        if current_vel_norm > EPSILON:
             action_space_with_current = possible_actions + [robot_vel]
        else:
             action_space_with_current = possible_actions

        # Select robot action using the algorithm
        selected_action = select_social_momentum_action(
            robot_q=robot_pos,
            current_robot_velocity=robot_vel,
            robot_goal_q=robot_goal,
            all_humans_q=[human_pos], # Pass current positions as list
            all_humans_v=[human_vel], # Pass current velocities as list
            robot_action_space=action_space_with_current,
            lambda_sm=SOCIAL_MOMENTUM_WEIGHT,
            time_step=SIM_TIME_STEP,
            robot_radius=DEFAULT_ROBOT_RADIUS,
            human_radius=DEFAULT_HUMAN_RADIUS,
            fov_deg=DEFAULT_FOV_DEG
        )

        # Update robot velocity based on selected action
        robot_vel = selected_action

        # Update positions (simple Euler integration)
        robot_pos += robot_vel * SIM_TIME_STEP
        human_pos += human_vel * SIM_TIME_STEP # Human keeps constant velocity for this example

        # --- Hallway Boundary Collision (Simple Bounce) ---
        # Robot
        if not (-HALLWAY_WIDTH / 2 < robot_pos[0] < HALLWAY_WIDTH / 2):
            robot_vel[0] *= -0.5 # Reverse x-velocity and dampen
            robot_pos[0] = np.clip(robot_pos[0], -HALLWAY_WIDTH / 2, HALLWAY_WIDTH / 2) # Keep inside
        # Human
        if not (-HALLWAY_WIDTH / 2 < human_pos[0] < HALLWAY_WIDTH / 2):
            human_vel[0] *= -0.5 # Reverse x-velocity and dampen
            human_pos[0] = np.clip(human_pos[0], -HALLWAY_WIDTH / 2, HALLWAY_WIDTH / 2)

        # Keep track of trajectories
        robot_traj_x.append(robot_pos[0])
        robot_traj_y.append(robot_pos[1])
        human_traj_x.append(human_pos[0])
        human_traj_y.append(human_pos[1])

        # Update plot elements
        robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
        human_dot.set_data([human_pos[0]], [human_pos[1]])
        trail_line_r.set_data(robot_traj_x, robot_traj_y)
        trail_line_h.set_data(human_traj_x, human_traj_y)
        time_text.set_text(f'Time: {current_time:.1f}s')

        # Increment time
        current_time += SIM_TIME_STEP

        # Check if robot reached goal (optional stop condition)
        if np.linalg.norm(robot_pos - robot_goal) < DEFAULT_ROBOT_RADIUS:
            print("Robot reached goal!")
            ani.event_source.stop()

        return robot_dot, human_dot, time_text, trail_line_r, trail_line_h # Return updated artists for blitting

    # --- Run Animation ---
    num_frames = int(TOTAL_SIM_TIME / SIM_TIME_STEP)
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=SIM_TIME_STEP * 1000, # Interval in milliseconds
                                  blit=True, repeat=False)

    plt.show()

    print(f"\nSimulation finished. Final Robot Pos: {robot_pos}, Final Human Pos: {human_pos}")