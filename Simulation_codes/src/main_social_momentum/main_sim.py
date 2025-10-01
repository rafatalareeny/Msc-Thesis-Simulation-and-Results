# main_sim.py
"""
Main entry point for running the Social Momentum simulation with visualization.
Uses a SimulationRunner class to manage state and logic.
"""
import numpy as np
import argparse
import sys
from typing import Dict, List

# Import modules
import environment as env
import social_momentum as sm
import visualisation as vis
import geometry_utils as geo

# --- Simulation Control Parameters ---
TOTAL_SIM_TIME = 20.0 # Longer time for teleop might be useful
SIM_TIME_STEP = env.DEFAULT_TIME_STEP

# --- Algorithm Parameters ---
SOCIAL_MOMENTUM_WEIGHT = 1
ROBOT_FOV_DEG = sm.DEFAULT_FOV_DEG
ROBOT_RADIUS = env.DEFAULT_ROBOT_RADIUS
HUMAN_RADIUS = env.DEFAULT_HUMAN_RADIUS

# --- Simulation Runner Class ---
class SimulationRunner:
    def __init__(self, args):
        """Initialize the SimulationRunner with command line arguments."""
        self.args = args
        self.setup_simulation()

    def setup_simulation(self):
        """Initializes simulation state based on args."""
        print(f"Selected Mode: {self.args.mode}")
        self.mode = self.args.mode
        self.plot_title = f'Social Momentum (λ={SOCIAL_MOMENTUM_WEIGHT}) - {self.mode.replace("_"," ").title()}'

        # Initialize agent variables
        self.robot_pos, self.robot_vel, self.robot_goal = None, None, None
        self.human_positions, self.human_velocities = [], []

        if self.mode == 'default':
            self.robot_pos, self.robot_vel, self.robot_goal = env.init_robot_default()
            human_pos, human_vel = env.init_human_default()
            self.human_positions = [human_pos]
            self.human_velocities = [human_vel]
            self.plot_title += " (1 Human)"
        elif self.mode == 'multiagent_mode':
            if self.args.num_humans < 1:
                print("Error: Number of humans for multiagent_mode must be at least 1.")
                sys.exit(1)
            print(f"Initializing {self.args.num_humans} human agent(s).")
            self.robot_pos, self.robot_vel, self.robot_goal = env.init_robot_random_goal()
            # Need robot_pos for spawn check
            if self.robot_pos is None:
                 # Fallback if init failed somehow, though unlikely
                 self.robot_pos = np.array([0.0, 0.0])
            self.human_positions, self.human_velocities = env.init_multiple_humans_random(
                num_humans=self.args.num_humans, robot_start_pos=self.robot_pos )
            self.plot_title += f" ({self.args.num_humans} Humans)"
        elif self.mode == 'teleop':
            print("Initializing Teleop Mode...")
            self.robot_pos, self.robot_vel, self.robot_goal = env.init_robot_default()
            human_pos, human_vel = env.init_human_teleop()
            self.human_positions = [human_pos]
            self.human_velocities = [human_vel]
            self.plot_title += " (Teleop Control)"
        elif self.mode == 'ionc':
    		# Robot starts at (1, 5) going to (8, 5)
            self.robot_pos, self.robot_vel, self.robot_goal = [1.0, 5.0], [0.0, 0.0], [8.0, 5.0]
    		# Human is placed directly in path, facing the robot (via velocity)
            self.human_positions = [[4.0, 5.0]]
            self.human_velocities = [[-0.1, 0.0]]  # Facing toward robot (simulated)

        else:
            print(f"Error: Unknown mode '{self.mode}'")
            sys.exit(1)

        # Common parameters stored as attributes
        self.robot_action_space = self.create_robot_action_space(env.ROBOT_MAX_SPEED)
        self.current_time = 0.0
        self.time_step = SIM_TIME_STEP
        self.lambda_sm = SOCIAL_MOMENTUM_WEIGHT
        self.robot_radius = ROBOT_RADIUS
        self.human_radius = HUMAN_RADIUS
        self.fov = ROBOT_FOV_DEG
        self.hallway_width = env.HALLWAY_WIDTH
        self.plot_length = env.PLOT_LENGTH
        self.max_time = TOTAL_SIM_TIME
        self.goal_threshold_sq = (self.robot_radius + 0.1)**2

    def create_robot_action_space(self, max_speed: float) -> List[np.ndarray]:
        """Creates the discrete set of possible velocity actions for the robot."""
        actions = [np.array([0.0, 0.0])]
        angles = np.linspace(-np.pi / 4, np.pi / 4, 5)
        speeds = [max_speed * 0.5, max_speed]
        for speed in speeds:
            for angle in angles:
                action = np.array([speed * np.sin(angle), speed * np.cos(angle)])
                actions.append(action)
        return actions

    def step(self) -> bool:
        """
        Performs one step of the simulation logic.
        Updates the instance's state attributes.

        Returns:
            bool: True if the simulation should continue, False otherwise.
        """

        # --- Termination Checks ---
        if self.current_time >= self.max_time:
            print(f"\nSimulation time limit ({self.max_time:.1f}s) reached.")
            return False
        # Ensure robot_pos and robot_goal are valid before distance check
        if self.robot_pos is None or self.robot_goal is None:
            print("Error: Robot position or goal not initialized.")
            return False
        dist_sq_to_goal = np.sum((self.robot_pos - self.robot_goal)**2)
        if dist_sq_to_goal < self.goal_threshold_sq:
            print("\nRobot reached goal!")
            return False

        # --- Robot Action Selection ---
        action_space_current = list(self.robot_action_space)
        if self.robot_vel is not None and np.linalg.norm(self.robot_vel) > geo.EPSILON:
            action_space_current.append(self.robot_vel)

        # Ensure all inputs to select_social_momentum_action are valid
        if self.robot_pos is None or self.robot_vel is None or self.robot_goal is None:
             print("Error: Robot state incomplete for action selection.")
             return False # Stop simulation if state is bad


        selected_robot_action = sm.select_social_momentum_action(
            robot_q=self.robot_pos,
            current_robot_velocity=self.robot_vel,
            robot_goal_q=self.robot_goal,
            all_humans_q=self.human_positions,
            all_humans_v=self.human_velocities,
            robot_action_space=action_space_current,
            lambda_sm=self.lambda_sm,
            time_step=self.time_step,
            robot_radius=self.robot_radius,
            human_radius=self.human_radius,
            fov_deg=self.fov
        )

        # --- Update Agent States ---
        # Robot Update
        new_robot_pos, new_robot_vel_after_update = env.update_agent_state(
            pos=self.robot_pos,
            vel=selected_robot_action, # Apply the selected action
            time_step=self.time_step,
            hallway_width=self.hallway_width
        )

        # Human State Update
        new_human_positions = []
        new_human_velocities_after_update = []
        if self.mode == 'teleop':
            # Ensure there's a human to control
            if not self.human_positions:
                 print("Warning: Teleop mode active but no human agents exist.")
            else:
                teleop_vel = vis.get_teleop_velocity()
                h_pos = self.human_positions[0]
                new_h_pos, _ = env.update_agent_state(
                    pos=h_pos, vel=teleop_vel, time_step=self.time_step, hallway_width=self.hallway_width
                )
                new_human_positions.append(new_h_pos)
                # Store the velocity command that was used for the update
                new_human_velocities_after_update.append(teleop_vel)
        else: # Default or multiagent
            for h_pos, h_vel in zip(self.human_positions, self.human_velocities):
                new_h_pos, current_h_vel_after_update = env.update_agent_state(
                    pos=h_pos, vel=h_vel, time_step=self.time_step, hallway_width=self.hallway_width
                )
                new_human_positions.append(new_h_pos)
                # Store the velocity *after* potential wall collisions
                new_human_velocities_after_update.append(current_h_vel_after_update)

        # --- Update internal state attributes ---
        self.robot_pos = new_robot_pos
        # Store the velocity that was *applied* for this step (the selected action)
        self.robot_vel = selected_robot_action # Or new_robot_vel_after_update if you want resulting vel
        self.human_positions = new_human_positions
        self.human_velocities = new_human_velocities_after_update # Store resulting velocities
        self.current_time += self.time_step

        return True # Continue simulation

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Social Momentum Simulation")
    parser.add_argument( 'mode', nargs='?', default='default',
        choices=['default', 'multiagent_mode', 'teleop'],
        help="Simulation mode ('default', 'multiagent_mode', 'teleop')" )
    parser.add_argument( 'num_humans', type=int, nargs='?', default=1,
        help="Number of human agents (used in multiagent_mode)" )
    args = parser.parse_args()

    # Create the simulation runner instance - this calls setup_simulation
    try:
        runner = SimulationRunner(args)
    except Exception as e:
        print(f"Error during SimulationRunner initialization: {e}")
        sys.exit(1)


    # --- Setup Visualization ---
    # Ensure runner has valid state before setting up plot
    if runner.robot_goal is None:
         print("Error: Robot goal not set after initialization. Cannot setup plot.")
         sys.exit(1)

    print("Setting up visualization...")
    vis.setup_plot(
        hallway_width=runner.hallway_width,
        plot_length=runner.plot_length,
        robot_goal_pos=runner.robot_goal,
        human_max_speed=env.HUMAN_MAX_SPEED, # Still needed for teleop speed setting
        mode=runner.mode,
        title=runner.plot_title
    )

    # --- Run Simulation via Animation ---
    print("Running simulation...")
    # Pass the runner instance itself to the animation function
    # The animation function will call runner.step() internally
    vis.run_simulation_animation(
        total_sim_time=runner.max_time,
        sim_time_step=runner.time_step,
        # Pass the runner instance which has the step method and state
        runner_instance=runner
    )

    print(f"\nSimulation finished.")
    # Access final state from the runner instance
    print(f"Final Robot Pos: {runner.robot_pos}")
    if len(runner.human_positions) == 1:
        print(f"Final Human Pos: {runner.human_positions[0]}")
    else:
         print(f"Final Human Positions: {runner.human_positions}")
