#!/usr/bin/env python3
"""
run_intent_evaluation_with_logging.py

- Adds per-trial logging of features (width, gaze_angle, body_angle, speed_label, ground_truth)
  into CSV for Bayesian network CPT estimation.
- Retains proximity-only filtering for evaluation.
"""
import sys, math, random
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# Ensure SM code is importable
SM_PATH = "/home/ralareeny/social_momentum/src"
if SM_PATH not in sys.path:
    sys.path.append(SM_PATH)

from main_social_momentum.environment import update_agent_state, HUMAN_MAX_SPEED, PLOT_LENGTH, DEFAULT_TIME_STEP
from main_social_momentum.social_momentum import select_social_momentum_action

# Parameters of the simulation
"""
You can adjust the parameters of the simulation which includes the length of the corridor, robot and human actor raduis, and the proximity threshold. 

The proximity thereshold dectates the displacement between the robot and the person where the robot stores the other variables

The variables logged when a person is in the robot proximity therershold are the velocity classification, relative gaze angle, relative body orientation agnle, and intent.

Other variables including the human max speed, robot speed and the lambda "social momentum coefficient are better to be changed from the main code file (social_momentum.py)
"""
CORRIDOR_LENGTH   = PLOT_LENGTH
ROBOT_RADIUS      = 0.25
ROBOT_SPEED       = 0.005
HUMAN_RADIUS      = 0.25
PROXIMITY_THRESH  = 3.0
TIME_STEP         = DEFAULT_TIME_STEP
HUMAN_SPEED       = HUMAN_MAX_SPEED

# Angle thresholds
"""
The theresholds are based on the state of the art frameworks analysed during the implementation and domain experts. The body angle thereshold is abstracted from Pal robotics.

Pal Robotics, have implemented the "Detect people oriented toward the robot" node as part of the "hri_body_detect" pipeline. 

In there approach they classify a person facing if the angle is within 20 degrees. 

With regards to the gaze, we simulate different gaze behavior to differentiate when the actor is intentionally obstructing or just navigating around the robot. 
Intentional states have more consice directed gaze while in normal navigation it can reflection environment scanning whether it its looking at the robot or pass it. 
"""
GAZE_OBSTRUCT_RAD = math.radians(15.0)
GAZE_AVOID_RAD    = math.radians(25.0)
ANGLE_THRESH_BODY = math.radians(20.0)

# Build action space
HUMAN_ACTION_SPACE = []
for s in (0.5, 1.0):
    for α in np.linspace(-math.pi/4, math.pi/4, 180):
        HUMAN_ACTION_SPACE.append(np.array([
            s*HUMAN_SPEED*math.sin(α),
           -s*HUMAN_SPEED*math.cos(α)
        ]))
HUMAN_ACTION_SPACE.append(np.zeros(2))

PAUSE_TIME = 0.0  # no pause

# Utilities  (refer to the main social momentum code in the the "social_momentum_mail.py")
def compute_angle(v1, v2):
    if np.linalg.norm(v1)<1e-6 or np.linalg.norm(v2)<1e-6:
        return math.pi
    u1, u2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
    return math.acos(np.clip(np.dot(u1,u2), -1.0,1.0))

# Simulation components (refer to the main social momentum code in the the "social_momentum_mail.py")
"""
we have here 3 main function that controls the simulation. 

update_robot: move along x axis and bounce off walls. This will create variability in the robot position. 

speed_scaler: linear slow-down near the goal (0→1). This is aimed to mimic how humans slow down when reaching their goal. 

update_human: pick social momentum action vs robot, step state, cap/taper speed. these are all based on the original social momentum model original code. 
refer to the main_social_momentum model.py for more illustrative code. 
"""

def update_robot(r_pos, r_vel, w):
    x,y = r_pos; vx,vy = r_vel
    x_new = x + vx*TIME_STEP
    if abs(x_new)+ROBOT_RADIUS > w/2:
        vx = -vx; x_new = x + vx*TIME_STEP
    r_pos[:] = (x_new, y); r_vel[:] = (vx, vy)
    return r_pos, r_vel

def speed_scaler(dist, ramp=5.0):
    return max(0.0, min(1.0, dist / ramp))

def update_human(h_pos, h_vel, h_goal, r_pos, r_vel, w, λ):
    v_cand = select_social_momentum_action(
        robot_q=h_pos, current_robot_velocity=h_vel,
        robot_goal_q=h_goal,
        all_humans_q=[r_pos], all_humans_v=[r_vel],
        robot_action_space=HUMAN_ACTION_SPACE,
        lambda_sm=λ, time_step=TIME_STEP,
        robot_radius=HUMAN_RADIUS, human_radius=ROBOT_RADIUS
    )
    new_pos, new_vel = update_agent_state(h_pos, v_cand, TIME_STEP, w)
    # ramp down near goal
    d_goal = np.linalg.norm(h_goal - new_pos)
    scale = speed_scaler(d_goal)
    if np.linalg.norm(new_vel)>1e-6:
        new_vel = (new_vel/np.linalg.norm(new_vel)) * (HUMAN_SPEED * scale)
    return new_pos, new_vel

# Single trial with feature logging 
"""
Controlling the runs of each trial in the simulation. Three main objectives: 

Spawn robot/human: set human goal & λ by mode (obstruct→target robot, avoid→exit).

Step both until proximity hit or human exits.

On proximity (if Ture): compute body_angle, gaze_angle (with jitter by gaze_rad), speed_label, and return with ground_truth (1=obstruct, 0=avoid). Else return Nones.
"""
def run_trial(width, mode):
    # init robot
    xr = random.uniform(-(width/2-ROBOT_RADIUS),(width/2-ROBOT_RADIUS))
    r_pos = np.array([xr, CORRIDOR_LENGTH/2], float)
    r_vel = np.array([random.choice([-ROBOT_SPEED,ROBOT_SPEED]), 0.0], float)
    # init human
    xh = random.uniform(-(width/2-HUMAN_RADIUS),(width/2-HUMAN_RADIUS))
    h_pos = np.array([xh, CORRIDOR_LENGTH-0.1], float)
    h_vel = np.array([0.0, -HUMAN_SPEED], float)
    # mode-specific
    if mode=='obstruct':
        h_goal, λ, gt = r_pos.copy(), 0.0, 1
        gaze_rad = GAZE_OBSTRUCT_RAD
    else:
        xg = random.uniform(-(width/2-HUMAN_RADIUS),(width/2-HUMAN_RADIUS))
        h_goal, λ, gt = np.array([xg,0.0],float), 0.01, 0
        gaze_rad = GAZE_AVOID_RAD
    max_steps = int((CORRIDOR_LENGTH/(HUMAN_SPEED*TIME_STEP))*5)
    for _ in range(max_steps):
        r_pos, r_vel = update_robot(r_pos, r_vel, width)
        h_pos, h_vel = update_human(h_pos, h_vel, h_goal, r_pos, r_vel, width, λ)
        dist = np.linalg.norm(r_pos - h_pos)
        if dist <= PROXIMITY_THRESH:
            # features
            body_ang = compute_angle(h_vel, r_pos - h_pos)
            base = math.atan2(h_vel[1], h_vel[0])
            δ = random.uniform(-gaze_rad, gaze_rad)
            gaze_ang = compute_angle(np.array([math.cos(base+δ),math.sin(base+δ)]), r_pos - h_pos)
            speed = np.linalg.norm(h_vel)
            speed_label = 'fast' if speed>0.75*HUMAN_SPEED else ('slow' if speed>1e-6 else 'stopped')
            return gt, body_ang, gaze_ang, speed_label
        if h_pos[1] <= 0.1:
            return None, None, None, None
    return None, None, None, None

# Run experiments for the simulation data generation
widths  = [1.0, 1.5, 2.0, 3.0]
trials  = 40
records = []
for w in widths:
    for mode in ('obstruct','avoid'):
        for _ in range(trials):
            gt, body_ang, gaze_ang, speed = run_trial(w, mode)
            if gt is not None:
                records.append({
                    'width': w,
                    'mode': mode,
                    'ground_truth': gt,
                    'body_angle': math.degrees(body_ang),
                    'gaze_angle': math.degrees(gaze_ang),
                })
# save the dataFrame and CSV
df = pd.DataFrame(records)
df.to_csv('Simulation_dataset.csv', index=False)

print("Saved Simulation_dataset.csv")

