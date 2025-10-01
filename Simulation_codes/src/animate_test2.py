#!/usr/bin/env python3

import sys, math, random, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

SM_PATH = "/home/ralareeny/social_momentum/src"
if SM_PATH not in sys.path:
    sys.path.append(SM_PATH)

from main_social_momentum.environment import update_agent_state, HUMAN_MAX_SPEED, PLOT_LENGTH, DEFAULT_TIME_STEP
from main_social_momentum.social_momentum import select_social_momentum_action

# Parameters of the sim/snimation
"""
You can adjust the parameters of the simulation which includes the length of the corridor, robot and human actor raduis, and the proximity threshold. 

The proximity thereshold dectates the displacement between the robot and the person where the robot stores the other variables

The variables logged when a person is in the robot proximity therershold are the velocity classification, relative gaze angle, relative body orientation agnle, and intent.

Other variables including the human max speed, robot speed and the lambda "social momentum coefficient are better to be changed from the main code file (social_momentum.py)
"""
CORRIDOR_LENGTH   = PLOT_LENGTH    
ROBOT_RADIUS      = 0.25
ROBOT_SPEED       = 0.01            
HUMAN_RADIUS      = 0.25

PROXIMITY_THRESH  = 2.0           
ANG_THRESH_BODY   = math.radians(15.0)  
TIME_STEP         = DEFAULT_TIME_STEP
HUMAN_SPEED       = HUMAN_MAX_SPEED

# Gaze offset ranges per mode
GAZE_OFFSET = {
    'obstruct': math.radians(10.0),
    'avoid':    math.radians(25.0)
}

# Discrete action space for the social momentum human planner
"""
Here we define the action space in which the human actor can operate. This is mainly based on the original code obtained from the social momentum model. 
"""
HUMAN_ACTION_SPACE = []
for s in (0.5, 1.0):
    for alpha in np.linspace(-math.pi/4, math.pi/4, 180):
        HUMAN_ACTION_SPACE.append(np.array([s*HUMAN_SPEED*math.sin(alpha),
                                           -s*HUMAN_SPEED*math.cos(alpha)]))
HUMAN_ACTION_SPACE.append(np.zeros(2))

PAUSE_TIME = 0.05  # s/frame

# utility function 
"""
Return the angle (radians) between two vectors (robot and human).
"""
def compute_angle(v1, v2):
    if np.linalg.norm(v1)<1e-6 or np.linalg.norm(v2)<1e-6:
        return math.pi
    u1, u2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
    return math.acos(np.clip(np.dot(u1,u2), -1.0,1.0))

# Update function
"""
we have here 3 main function that controls the simulation. 

update_robot: move along x axis and bounce off walls. This will create variability in the robot position. 

speed_scaler: linear slow-down near the goal (0→1). This is aimed to mimic how humans slow down when reaching their goal. 

update_human: pick social momentum action vs robot, step state, cap/taper speed. these are all based on the original social momentum model original code. 
refer to the main_social_momentum model.py for more illustrative code. 
"""
def update_robot(pos, vel, width):
    x,y = pos; vx,vy = vel
    x_new = x + vx*TIME_STEP
    if abs(x_new) + ROBOT_RADIUS > width/2:
        vx = -vx
        x_new = x + vx*TIME_STEP
    pos[:] = (x_new, y)
    vel[:] = (vx, vy)
    return pos, vel


def update_human(pos, vel, goal, r_pos, r_vel, width, lambda_sm):
    v_cand = select_social_momentum_action(
        robot_q=pos, current_robot_velocity=vel,
        robot_goal_q=goal,
        all_humans_q=[r_pos], all_humans_v=[r_vel],
        robot_action_space=HUMAN_ACTION_SPACE,
        lambda_sm=lambda_sm, time_step=TIME_STEP,
        robot_radius=HUMAN_RADIUS, human_radius=ROBOT_RADIUS
    )
    new_pos, new_vel = update_agent_state(pos, v_cand, TIME_STEP, width)
    
    # ramp speed down within 3m of goal
    
    dist_goal = np.linalg.norm(goal - new_pos)
    scale = max(0.0, min(1.0, dist_goal/1.0))
    if np.linalg.norm(new_vel) > 1e-6:
        dir = new_vel / np.linalg.norm(new_vel)
        new_vel = dir * (HUMAN_SPEED * scale)
    return new_pos, new_vel

# Animate trial
"""
This function is based on the simulation code. the main objective is to run and visualize a single simulation trial: 

initializes robot/human, updates them each frame, draws their motion,

when they get close (proximity condition is met =True) it infers/prints intent, then stops (or stops if the human reaches the goal).
"""
def animate_trial(width, mode):
    # robot init
    xr = random.uniform(-(width/2 - ROBOT_RADIUS), (width/2 - ROBOT_RADIUS))
    r_pos = np.array([xr, CORRIDOR_LENGTH/2], float)
    r_vel = np.array([random.choice([-ROBOT_SPEED, ROBOT_SPEED]), 0.0], float)

    # human init
    xh = random.uniform(-(width/2 - HUMAN_RADIUS), (width/2 - HUMAN_RADIUS))
    h_pos = np.array([xh, CORRIDOR_LENGTH - 0.1], float)
    if mode == 'obstruct':
        h_goal = r_pos.copy(); lambda_sm = 0.0
    else:
        xg = random.uniform(-(width/2 - HUMAN_RADIUS), (width/2 - HUMAN_RADIUS))
        h_goal = np.array([xg, 0.0], float); lambda_sm = 0.01
    h_vel = np.array([0.0, -HUMAN_SPEED], float)

    fig, ax = plt.subplots(figsize=(5, 8)); plt.ion()
    ax.set_xlim(-width/2 - 0.2, width/2 + 0.2)
    ax.set_ylim(-0.5, CORRIDOR_LENGTH + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"W={width:.1f}m, mode={mode}")
    ax.add_patch(Rectangle((-width/2, 0), 0.02, CORRIDOR_LENGTH, color='black'))
    ax.add_patch(Rectangle(( width/2 - 0.02, 0), 0.02, CORRIDOR_LENGTH, color='black'))
    ax.plot([-width/2, width/2], [0, 0], color='green')
    r_patch = Circle(r_pos, ROBOT_RADIUS, color='red');   ax.add_patch(r_patch)
    h_patch = Circle(h_pos, HUMAN_RADIUS, color='blue');  ax.add_patch(h_patch)

    vel_arr = orient_arr = los_line = None
    max_steps = int((CORRIDOR_LENGTH / (HUMAN_SPEED * TIME_STEP)) * 5)

    gaze_range = GAZE_OFFSET[mode]

    for step in range(max_steps):
        # update robot & human
        r_pos, r_vel = update_robot(r_pos, r_vel, width)
        r_patch.center = tuple(r_pos)
        h_pos, h_vel = update_human(h_pos, h_vel, h_goal, r_pos, r_vel, width, lambda_sm)
        h_patch.center = tuple(h_pos)

        # clear old arrows/lines
        for artist in (vel_arr, orient_arr, los_line):
            if artist:
                artist.remove()

        # draw body arrow
        speed = np.linalg.norm(h_vel)
        label = 'stopped' if speed < 1e-6 else ('fast' if speed > 0.75*HUMAN_SPEED else 'slow')
        if speed > 1e-6:
            u = h_vel / speed
            vel_arr = ax.arrow(*h_pos, *(u * 0.7), head_width=0.05, head_length=0.05,
                               fc='blue', ec='blue')

        # draw gaze arrow (offset based on mode)
        if speed > 1e-6:
            base_ang = math.atan2(h_vel[1], h_vel[0])
            δ = random.uniform(-gaze_range, gaze_range)
            face_ang = base_ang + δ
            face_dir = np.array([math.cos(face_ang), math.sin(face_ang)])
            orient_arr = ax.arrow(*h_pos, *(face_dir * 0.7), head_width=0.05, head_length=0.05,
                                   fc='green', ec='green')
        else:
            face_dir = np.zeros(2)
            orient_arr = None

        plt.pause(PAUSE_TIME)

        # proximity & intent (based on body orientation)
        d = np.linalg.norm(r_pos - h_pos)
        if d <= PROXIMITY_THRESH:
            body_ang = compute_angle(h_vel, r_pos - h_pos)
            intent = (body_ang <= ANG_THRESH_BODY)
            print(f"W={width:.1f} | body_ang={math.degrees(body_ang):5.1f}° | speed={label} | -> {intent}")
            los_line, = ax.plot([h_pos[0], r_pos[0]], [h_pos[1], r_pos[1]], '--', color='orange')
            plt.pause(0.5)
            break

        # goal reached
        if h_pos[1] <= 0.1:
            print(f"W={width:.1f}m | reached goal | speed={label}")
            break

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('obstruct','avoid'), default='avoid')
    args = parser.parse_args()
    for w in [3.0, 2.0, 1.5, 1.0]:
        animate_trial(w, args.mode)
