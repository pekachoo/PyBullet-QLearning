# greedy_demo.py
import pybullet as p, pybullet_data as pd, time, math, random
import numpy as np, json

# --- load artifacts ---
QTable = np.load("qtable.npy")
with open("qtable_meta.json") as f:
    M = json.load(f)

offset_interval = M["offset_interval"]
y_max = M["y_max"]
yaw_interval = M["yaw_interval"]
num_y_bins = M["num_y_bins"]
num_yaw_bins = M["num_yaw_bins"]
num_actions = M["num_actions"]

# --- env constants (must match training) ---
W = 10.0
L = 100.0
wall_th = 0.1
wall_h  = 1.0
dt = 1/240
speed = 2.0
START_WALL_OFFSET = 1.25

# --- setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, wall_th/2, wall_h/2])
wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L/2, wall_th/2, wall_h/2], rgbaColor=[0.8,0.8,0.8,1])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, baseVisualShapeIndex=wall_vis, basePosition=[0,  W/2, wall_h/2])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, baseVisualShapeIndex=wall_vis, basePosition=[0, -W/2, wall_h/2])

box = p.loadURDF("/urdf/simple_box.urdf", [0,0,0.5])
p.changeDynamics(box, -1, lateralFriction=0.25, rollingFriction=0.05, spinningFriction=0.15)

def getCurrAngle(q):
    _,_,yaw = p.getEulerFromQuaternion(q)
    return yaw

def resetPos(obj, wall_offset):
    y_random = random.uniform(-(W/2 - wall_offset), (W/2 - wall_offset))
    yaw_random = random.uniform(-math.pi, math.pi)
    p.resetBasePositionAndOrientation(obj, [0, y_random, 0], p.getQuaternionFromEuler([0,0,yaw_random]))
    p.resetBaseVelocity(obj, [0,0,0], [0,0,0])

def y_to_bin_index(y, offset_interval, y_max):
    y_clamped = np.clip(y, -y_max, y_max)
    nearest = round((y_clamped + y_max) / offset_interval) * offset_interval
    idx = int(nearest / offset_interval)
    # safety clamp
    return max(0, min(idx, num_y_bins - 1))

def yaw_to_bin_index(yaw, yaw_interval):
    # yaw in [-pi, pi]
    yaw_shifted = yaw + math.pi
    nearest = round(yaw_shifted / yaw_interval) * yaw_interval
    idx = int(nearest / yaw_interval)
    return max(0, min(idx, num_yaw_bins - 1))

def state_to_bin_index(y, yaw):
    yb   = y_to_bin_index(y, offset_interval, y_max)
    yawb = yaw_to_bin_index(yaw, yaw_interval)
    return int(yb * num_yaw_bins + yawb)

# actions must match training order: 0=forward, 1=right, 2=left
def steer_right(obj): p.applyExternalTorque(obj, -1, [0,0, 20], p.LINK_FRAME)
def steer_left(obj):  p.applyExternalTorque(obj, -1, [0,0,-20], p.LINK_FRAME)
def go_forward(obj):  pass

def executeAction(a, obj):
    if a == 0: go_forward(obj)
    elif a == 1: steer_right(obj)
    elif a == 2: steer_left(obj)

def forward_push_from_yaw(yaw, speed):
    return speed*math.cos(yaw), speed*math.sin(yaw)

resetPos(box, START_WALL_OFFSET)
for t in range(10000):  # ~20s
    (_, y, _), q = p.getBasePositionAndOrientation(box)
    yaw = getCurrAngle(q)
    s = state_to_bin_index(y, yaw)
    a = int(np.argmax(QTable[s])) # greedy action
    executeAction(a, box)
    vx, vy = forward_push_from_yaw(yaw, speed)
    p.resetBaseVelocity(box, [vx, vy, 0])
    p.stepSimulation()
    time.sleep(dt)
