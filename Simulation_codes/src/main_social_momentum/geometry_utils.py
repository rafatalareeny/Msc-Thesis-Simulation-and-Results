# geometry_utils.py
"""
Module for mathematical and geometric helper functions used in the simulation.
"""
import numpy as np
      
from typing import Tuple

    

EPSILON = 1e-6  # Small number to avoid division by zero

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
        The Z-component of the total angular momentum. Returns 0.0 if inputs
        are None or agents are at the same position.
    """
    if q1 is None or v1 is None or q2 is None or v2 is None:
        return 0.0
    # Ensure inputs are numpy arrays
    q1 = np.asarray(q1)
    v1 = np.asarray(v1)
    q2 = np.asarray(q2)
    v2 = np.asarray(v2)

    if np.array_equal(q1, q2): # Avoid issues if agents are at the same spot
        return 0.0

    pc = (q1 + q2) / 2.0  # Center of mass
    p1c = q1 - pc        # Position relative to center of mass
    p2c = q2 - pc

    # 2D cross product: Lz = x*vy - y*vx
    L1_z = p1c[0] * v1[1] - p1c[1] * v1[0]
    L2_z = p2c[0] * v2[1] - p2c[1] * v2[0]

    return L1_z + L2_z

def check_collision(robot_q: np.ndarray, robot_v_or_action: np.ndarray,
                    human_q: np.ndarray, human_v: np.ndarray,
                    time_step: float, robot_radius: float, human_radius: float) -> bool:
    """
    Checks for predicted collision between robot and one human in the next time step.

    Args:
        robot_q: Current robot position [x, y].
        robot_v_or_action: Proposed robot velocity (action) [vx, vy].
        human_q: Current human position [x, y].
        human_v: Current human velocity [vx, vy].
        time_step: Prediction time step.
        robot_radius: Robot collision radius.
        human_radius: Human collision radius.

    Returns:
        True if a collision is predicted, False otherwise.
    """
    robot_q_next = np.asarray(robot_q) + np.asarray(robot_v_or_action) * time_step
    human_q_next = np.asarray(human_q) + np.asarray(human_v) * time_step

    min_dist_sq = (robot_radius + human_radius)**2
    dist_sq = np.sum((robot_q_next - human_q_next)**2)

    # Use EPSILON to avoid floating point issues exactly at contact
    return dist_sq < min_dist_sq - EPSILON

def normalize(vector: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalizes a vector and returns the normalized vector and its original magnitude."""
    norm = np.linalg.norm(vector)
    if norm < EPSILON:
        return vector, 0.0 # Return zero vector and zero norm
    return vector / norm, norm