# # environment.py
# """
# Module defining the simulation environment, agent properties,
# initialization, and physics updates (position, boundaries).
# """
# import numpy as np
# import random
# from typing import List, Tuple

# # --- Environment Constants ---
# HALLWAY_WIDTH = 10.0   # meters # Make sure this matches your desired value
# PLOT_LENGTH = 15.0    # meters # Make sure this matches your desired value

# # --- Agent Default Properties ---
# DEFAULT_ROBOT_RADIUS = 0.3   # meters # Make sure this matches your desired value
# DEFAULT_HUMAN_RADIUS = 0.3   # meters # Make sure this matches your desired value
# ROBOT_MAX_SPEED = 1.2 # meters/second
# HUMAN_MAX_SPEED = 1.0 # meters/second
# DEFAULT_TIME_STEP = 0.1 # Simulation/Prediction time step (s)

# # --- Helper ---
# EPSILON = 1e-6 # From geometry_utils, but useful here too for boundary checks

# def init_human(hallway_width: float = HALLWAY_WIDTH,
#                plot_length: float = PLOT_LENGTH,
#                max_speed: float = HUMAN_MAX_SPEED) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Initializes human state (position, velocity) based on environment dimensions.
#     Uses the specific non-random setup from the user's last code version.

#     Returns:
#         Tuple: (initial_pos, initial_vel)
#     """
#     center_x = 0.0
#     initial_pos = np.array([center_x, plot_length / 2]) # Specific start from user code
#     initial_vel = np.array([0.0, -max_speed]) # Moving straight down

#     # --- Alternative Random Initialization (commented out) ---
#     # human_start_x = random.uniform(-hallway_width / 2 * 0.8, hallway_width / 2 * 0.8)
#     # human_start_y = random.uniform(plot_length * 0.3, plot_length * 0.7)
#     # initial_pos = np.array([human_start_x, human_start_y])

#     # start_speed_h = random.uniform(0.5, max_speed)
#     # start_angle_h = random.uniform(0, 2 * np.pi) # Random direction
#     # initial_vel = np.array([start_speed_h * np.cos(start_angle_h), start_speed_h * np.sin(start_angle_h)])
#     # initial_vel[0] *= 0.3
#     # initial_vel = np.clip(initial_vel, -max_speed, max_speed)

#     return initial_pos, initial_vel


# def update_agent_state(pos: np.ndarray, vel: np.ndarray, time_step: float,
#                          hallway_width: float) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Updates a single agent's position and handles wall collisions.

#     Args:
#         pos: Current position [x, y].
#         vel: Current velocity [vx, vy].
#         time_step: Simulation time step.
#         hallway_width: Width of the hallway for boundary checks.

#     Returns:
#         Tuple: (new_pos, new_vel) after update and boundary check.
#     """
#     pos = np.asarray(pos)
#     vel = np.asarray(vel)

#     new_pos = pos + vel * time_step
#     new_vel = vel.copy() # Start with current velocity

#     # --- Hallway Boundary Collision (Simple Bounce) ---
#     half_width = hallway_width / 2
#     if not (-half_width < new_pos[0] < half_width):
#         new_vel[0] *= -0.5 # Reverse x-velocity and dampen
#         # Clamp position strictly within bounds after reversing velocity
#         new_pos[0] = np.clip(new_pos[0], -half_width + EPSILON, half_width - EPSILON)

#     # Note: No collision check for top/bottom boundaries in this version

#     return new_pos, new_vel# environment.py
"""
Module defining the simulation environment, agent properties,
initialization, and physics updates (position, boundaries).
"""
import numpy as np
import random
from typing import List, Tuple

# Import geometry utils for distance checks during spawn
import geometry_utils as geo

# --- Environment Constants ---
HALLWAY_WIDTH = 10.0   # meters
PLOT_LENGTH = 15.0    # meters

# --- Agent Default Properties ---
DEFAULT_ROBOT_RADIUS = 0.5   # meters
DEFAULT_HUMAN_RADIUS = 0.5   # meters
ROBOT_MAX_SPEED = 1.0 # meters/second
HUMAN_MAX_SPEED = 3.0 # meters/second
DEFAULT_TIME_STEP = 0.1 # Simulation/Prediction time step (s)

# --- Helper ---
EPSILON = 1e-6 # Re-defined here for local use

# --- Spawn Safety ---
MIN_SPAWN_SEPARATION = 0.5 # Minimum distance between spawned agents


def init_robot_default() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes robot state for the default mode (fixed start/goal, forward velocity).

    Returns:
        Tuple: (initial_pos, initial_vel, goal_pos)
    """
    center_x = 0.0
    robot_start_x = 0.01 # Specific start X from user code
    initial_pos = np.array([robot_start_x, 0.0])
    goal_pos = np.array([center_x, PLOT_LENGTH]) # Goal at center top
    initial_vel = np.array([0.0, ROBOT_MAX_SPEED]) # Start moving straight up
    return initial_pos, initial_vel, goal_pos

def init_robot_random_goal() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes robot state for multi-agent mode (fixed start, forward velocity, random goal).

    Returns:
        Tuple: (initial_pos, initial_vel, goal_pos)
    """
    robot_start_x = 1.0 # Keep fixed start X for now
    initial_pos = np.array([robot_start_x, 0.0])
    initial_vel = np.array([0.0, ROBOT_MAX_SPEED]) # Start moving straight up

    # Random goal near the top edge of the hallway
    goal_x = random.uniform(-HALLWAY_WIDTH / 2 * 0.9, HALLWAY_WIDTH / 2 * 0.9)
    goal_y = random.uniform(PLOT_LENGTH * 0.8, PLOT_LENGTH * 0.95)
    goal_pos = np.array([goal_x, goal_y])

    return initial_pos, initial_vel, goal_pos

def init_human_default() -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes human state for the default mode (fixed start/velocity).

    Returns:
        Tuple: (initial_pos, initial_vel)
    """
    center_x = 0.0
    initial_pos = np.array([center_x, PLOT_LENGTH / 2]) # Specific start
    initial_vel = np.array([0.0, -HUMAN_MAX_SPEED]) # Moving straight down
    return initial_pos, initial_vel

def init_human_teleop() -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes human state for the teleop mode (top center, stationary).

    Returns:
        Tuple: (initial_pos, initial_vel)
    """
    center_x = 0.0
    initial_pos = np.array([center_x, PLOT_LENGTH * 0.9]) # Start near top center
    initial_vel = np.array([0.0, 0.0]) # Start stationary
    return initial_pos, initial_vel

def _get_random_human_state(occupied_positions: List[np.ndarray], robot_start_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to generate a single random human state, avoiding collisions."""
    max_tries = 50
    for _ in range(max_tries):
        pos_x = random.uniform(-HALLWAY_WIDTH / 2 * 0.9, HALLWAY_WIDTH / 2 * 0.9)
        # Spawn humans away from robot start and goal areas initially
        pos_y = random.uniform(PLOT_LENGTH * 0.2, PLOT_LENGTH * 0.8)
        pos = np.array([pos_x, pos_y])

        # Check distance to other already spawned humans and robot start
        too_close = False
        if np.linalg.norm(pos - robot_start_pos) < MIN_SPAWN_SEPARATION + DEFAULT_ROBOT_RADIUS + DEFAULT_HUMAN_RADIUS:
            too_close = True
        else:
            for other_pos in occupied_positions:
                if np.linalg.norm(pos - other_pos) < MIN_SPAWN_SEPARATION + 2 * DEFAULT_HUMAN_RADIUS:
                    too_close = True
                    break
        
        if not too_close:
            # Generate random velocity
            speed = random.uniform(0.3, HUMAN_MAX_SPEED)
            angle = random.uniform(0, 2 * np.pi)
            vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            # Optional: bias velocity slightly towards up/down?
            # vel[0] *= 0.5 # Reduce side-to-side tendency
            return pos, vel
            
    # Fallback if we can't find a good spot (should be rare)
    print("Warning: Could not find suitably spaced spawn location for a human.")
    pos = np.array([random.uniform(-HALLWAY_WIDTH/2, HALLWAY_WIDTH/2), 
                    random.uniform(PLOT_LENGTH*0.2, PLOT_LENGTH*0.8)])
    vel = np.array([0.0, 0.0]) 
    return pos, vel
    

def init_multiple_humans_random(num_humans: int, robot_start_pos: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Initializes a specified number of human agents with random positions and velocities,
    avoiding spawning too close to each other or the robot's start.

    Args:
        num_humans: The number of humans to initialize.
        robot_start_pos: The starting position of the robot to avoid spawning on top.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: List of initial positions, List of initial velocities.
    """
    human_positions = []
    human_velocities = []
    occupied = [] # Keep track of where we've already spawned

    for _ in range(num_humans):
        pos, vel = _get_random_human_state(occupied, robot_start_pos)
        human_positions.append(pos)
        human_velocities.append(vel)
        occupied.append(pos) # Add current pos to occupied list

    return human_positions, human_velocities


def update_agent_state(pos: np.ndarray, vel: np.ndarray, time_step: float,
                         hallway_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates a single agent's position and handles wall collisions.

    Args:
        pos: Current position [x, y].
        vel: Current velocity [vx, vy].
        time_step: Simulation time step.
        hallway_width: Width of the hallway for boundary checks.

    Returns:
        Tuple: (new_pos, new_vel) after update and boundary check.
    """
    pos = np.asarray(pos)
    vel = np.asarray(vel)

    new_pos = pos + vel * time_step
    new_vel = vel.copy() # Start with current velocity

    # --- Hallway Boundary Collision (Simple Bounce) ---
    half_width = hallway_width / 2
    if not (-half_width < new_pos[0] < half_width):
        new_vel[0] *= -0.5 # Reverse x-velocity and dampen
        new_pos[0] = np.clip(new_pos[0], -half_width + EPSILON, half_width - EPSILON)

    if not (0 < new_pos[1] < PLOT_LENGTH): # Keep human within plot vertically
        new_vel[1] *= -0.5 # Bounce vertically
        new_pos[1] = np.clip(new_pos[1], EPSILON, PLOT_LENGTH - EPSILON)

    return new_pos, new_vel