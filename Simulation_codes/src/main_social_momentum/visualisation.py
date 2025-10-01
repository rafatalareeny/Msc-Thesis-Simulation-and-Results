# visualisation.py
"""
Module for handling the simulation visualization using Matplotlib animation.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple, Dict, Callable, Any # Added Any

# Store plot elements globally within the module
fig, ax = None, None
robot_dot, goal_marker = None, None
time_text = None
ani = None # Animation object

# Store artists and trajectories for variable numbers of agents
human_dots: List[Any] = [] # List to hold human artist objects
robot_trail_line, human_trail_lines = None, [] # One robot trail, list for human trails
robot_traj_x, robot_traj_y = [], []
human_trajs_x: List[List[float]] = [] # List of lists for x trajectories
human_trajs_y: List[List[float]] = [] # List of lists for y trajectories

plot_artists: List[Any] = [] # Keep a list of all artists to return for blitting
plot_initialized = False
current_mode = 'default'

# This will hold the *target* velocity based on key presses
# It will be copied to the actual human velocity in main_sim.py's step function
teleop_target_velocity = np.array([0.0, 0.0])
teleop_speed = 1.0 # Use HUMAN_MAX_SPEED defined in environment.py

def on_key_press(event):
    """Handles key press events for teleoperation."""
    global teleop_target_velocity, teleop_speed
    # print(f"Key pressed: {event.key}") # Debug print

    if event.key == 'up':
        teleop_target_velocity = np.array([0.0, teleop_speed])
    elif event.key == 'down':
        teleop_target_velocity = np.array([0.0, -teleop_speed])
    elif event.key == 'left':
        teleop_target_velocity = np.array([-teleop_speed, 0.0])
    elif event.key == 'right':
        teleop_target_velocity = np.array([teleop_speed, 0.0])
    elif event.key in ['s', ' ']: # Stop key (s or space)
        teleop_target_velocity = np.array([0.0, 0.0])
    # Optional: Add diagonal movement or key release handling later if needed

def setup_plot(hallway_width: float, plot_length: float, robot_goal_pos: np.ndarray,
               human_max_speed: float, # Pass max speed for teleop
               mode: str, # <<< Pass the mode name here
               title: str = 'Social Momentum Simulation'):
    """
    Initializes the Matplotlib figure and axes for the simulation.
    """
    global fig, ax, robot_dot, goal_marker, time_text, robot_trail_line, plot_artists, plot_initialized
    global teleop_speed, teleop_target_velocity, current_mode # Access teleop variables & mode

    human_dots = []
    human_trail_lines = []
    robot_traj_x, robot_traj_y = [], []
    human_trajs_x = []
    human_trajs_y = []
    plot_artists = []
    plot_initialized = False
    teleop_target_velocity = np.array([0.0, 0.0]) # Reset target velocity
    teleop_speed = human_max_speed # Set teleop speed from environment

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(-hallway_width / 2 - 0.5, hallway_width / 2 + 0.5)
    ax.set_ylim(-1, plot_length)
    ax.set_aspect('equal', adjustable='box')
    # Adjust title based on mode
    plot_title_full = title
    if mode == 'teleop':
        plot_title_full += "\n(Use arrow keys for teleop, 's'/'space' to stop)"
    ax.set_title(plot_title_full)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)

    # --- Plot static elements ---
    wall1 = ax.axvline(-hallway_width / 2, color='k', linestyle='--', label='Hallway Wall')
    wall2 = ax.axvline(hallway_width / 2, color='k', linestyle='--')
    robot_dot, = ax.plot([], [], 'ro', markersize=8, label='Robot')
    goal_marker, = ax.plot([], [], 'gx', markersize=10, label='Goal') # Plot goal marker data later
    robot_trail_line, = ax.plot([], [], 'r:', alpha=0.5)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    goal_marker.set_data([robot_goal_pos[0]], [robot_goal_pos[1]])
    plot_artists.extend([robot_dot, goal_marker, time_text, robot_trail_line, wall1, wall2])
    ax.legend(loc='upper right')

    # --- Connect key handler ONLY if in teleop mode ---
    if mode == 'teleop':
        print("Teleop control enabled: Focus the plot window and use arrow keys.")
        # Store the handler ID in case we want to disconnect later (though removing explicit disconnect for now)
        # key_press_handler_id = fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_press_event', on_key_press)

def _update_plot(frame: int, runner): # runner is the SimulationRunner instance
    """ Internal function called by FuncAnimation for each frame. """
    global plot_initialized, human_dots, human_trail_lines, human_trajs_x, human_trajs_y, plot_artists, ani
    global robot_traj_x, robot_traj_y

    # --- Run one step using the runner's step method ---
    try:
        continue_sim = runner.step()
    except Exception as e:
        print(f"Error during runner.step(): {e}")
        # Optionally stop the animation here
        if ani:
            try: ani.event_source.stop()
            except AttributeError: pass
        return plot_artists # Return existing artists

    if not continue_sim:
        if ani:
            try: ani.event_source.stop()
            except AttributeError: pass
        return plot_artists

    # --- Get state directly from runner attributes ---
    # Use getattr for safety in case attributes somehow don't exist
    robot_pos = getattr(runner, 'robot_pos', np.array([0.,0.]))
    human_positions = getattr(runner, 'human_positions', [])
    current_time = getattr(runner, 'current_time', 0.0)
    mode = getattr(runner, 'mode', 'default')
    num_humans = len(human_positions)


    # --- Initialize human artists and trajectories on first *valid* frame ---
    if not plot_initialized and num_humans > 0:
        human_dots = []
        human_trail_lines = []
        human_trajs_x = [[] for _ in range(num_humans)]
        human_trajs_y = [[] for _ in range(num_humans)]
        new_artists = []
        for i in range(num_humans):
            label = f'Human {i+1}' if num_humans > 1 else 'Human'
            if mode == 'teleop':
                 label = 'Human (Teleop)'
            human_dot, = ax.plot([], [], 'bo', markersize=8, label=label)
            human_trail, = ax.plot([], [], 'b:', alpha=0.5)
            human_dots.append(human_dot)
            human_trail_lines.append(human_trail)
            new_artists.extend([human_dot, human_trail])

        plot_artists.extend(new_artists)
        try:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        except Exception:
             print("Warning: Could not update legend during initialization.")
             ax.legend(loc='upper right')

        plot_initialized = True
    elif num_humans == 0 and not plot_initialized:
        plot_initialized = True


    # --- Update trajectories ---
    robot_traj_x.append(robot_pos[0])
    robot_traj_y.append(robot_pos[1])
    for i in range(num_humans):
        # Check index bounds for safety
        if i < len(human_trajs_x) and i < len(human_positions):
            human_trajs_x[i].append(human_positions[i][0])
            human_trajs_y[i].append(human_positions[i][1])

    # --- Limit trail length ---
    max_trail_points = 100
    if len(robot_traj_x) > max_trail_points:
        robot_traj_x.pop(0)
        robot_traj_y.pop(0)
        for i in range(num_humans):
             if i < len(human_trajs_x) and len(human_trajs_x[i]) > max_trail_points:
                human_trajs_x[i].pop(0)
                human_trajs_y[i].pop(0)


    # --- Update plot element data ---
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    robot_trail_line.set_data(robot_traj_x, robot_traj_y)
    time_text.set_text(f'Time: {current_time:.1f}s')

    for i in range(min(num_humans, len(human_dots))):
        if i < len(human_positions):
            human_dots[i].set_data([human_positions[i][0]], [human_positions[i][1]])
            if i < len(human_trajs_x):
                 human_trail_lines[i].set_data(human_trajs_x[i], human_trajs_y[i])

    # Return list of *all* artists that might have been updated
    return plot_artists

# --- Modify run_simulation_animation to accept runner instance ---
def run_simulation_animation(total_sim_time: float, sim_time_step: float,
                             runner_instance): # Accept runner instance
    """ Runs the Matplotlib animation using a SimulationRunner instance. """
    global fig, ani

    if fig is None or ax is None:
        print("Error: Plot not setup. Call setup_plot first.")
        return

    # Use total_sim_time and sim_time_step from runner for consistency
    total_sim_time_actual = getattr(runner_instance, 'max_time', total_sim_time)
    sim_time_step_actual = getattr(runner_instance, 'time_step', sim_time_step)

    if sim_time_step_actual <= 0:
        print(f"Error: Invalid sim_time_step: {sim_time_step_actual}")
        return

    num_frames = int(total_sim_time_actual / sim_time_step_actual) + 1
    interval_ms = sim_time_step_actual * 1000

    # Pass the runner instance directly to _update_plot via fargs
    ani = animation.FuncAnimation(fig, _update_plot, frames=num_frames,
                                  fargs=(runner_instance,), # Pass runner instance in a tuple
                                  interval=interval_ms,
                                  blit=False, # Keep blit=False
                                  repeat=False, cache_frame_data=False)

    try:
        plt.show()
    except Exception as e:
        # Catch errors during the show() call, which might happen if the window is closed prematurely
        print(f"Exiting simulation window. Error during plt.show() (may be harmless): {e}")

    print("Animation finished or window closed.")

def get_teleop_velocity() -> np.ndarray:
    """Returns the current target velocity set by key presses."""
    global teleop_target_velocity
    return teleop_target_velocity.copy() # Return a copy

