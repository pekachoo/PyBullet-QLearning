import pybullet as p, pybullet_data as pd, time, math, random
import numpy as np
import matplotlib.pyplot as plt
import json

#constants

W = 10.0   # width
L = 100.0
wall_th = 0.1
wall_h = 1.0

magnitude = 15
left_force = magnitude
right_force = 5
increment = magnitude/100
forward = True
speed = 2.0

dt = 1/240
steps = int(100.0/dt)  #10 seconds total with dt steps/sec
message_interval = 5.0
next_interval = message_interval
current_time = 0

# y is in 0.25 intervals, yaw is in intervals of pi/6
y_max = 4
yaw_max = math.pi
offset_interval = 0.5
yaw_interval = 30 * math.pi/180
num_y_bins  = int(round(2*y_max/offset_interval)) + 1     # inclusive of both ends
num_yaw_bins = int(round(2*yaw_max/yaw_interval)) + 1     # inclusive of both ends
num_actions = 3
num_states  = num_y_bins * num_yaw_bins
QTable = np.zeros((num_states, num_actions))

# exploration constants
epsilon = 1.0


p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, wall_th/2, wall_h/2])
wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L/2, wall_th/2, wall_h/2], rgbaColor=[0.8,0.8,0.8,1])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, baseVisualShapeIndex=wall_vis,
                  basePosition=[0,  W/2, wall_h/2])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, baseVisualShapeIndex=wall_vis,
                  basePosition=[0, -W/2, wall_h/2])

box = p.loadURDF("/urdf/simple_box.urdf", [0,0,0.5])  # your box

# Reduce friction so it actually slides
p.changeDynamics(
    box, -1,
    lateralFriction=0.25,   # plastic on wood
    rollingFriction=0.05,   # lowkey irrelevant
    spinningFriction=0.15   # resist rotation
)

def getCurrAngle(quaternion):
    _, _, yaw = p.getEulerFromQuaternion(quaternion)
    return yaw

def PDCalculation(error, derivative, kp=65, kd=0.3):
    return kp*error + kd*derivative

def getIdealYaw(offset, lookahead=5):
    # cap it out at pi/2 and -pi/2
    if offset > lookahead:
        offset = lookahead
    elif offset < -lookahead:
        offset = -lookahead
    return -math.atan2(offset, math.sqrt(math.pow(lookahead, 2) - math.pow(offset, 2)))


#actions
def steer_right(obj):
    p.applyExternalTorque(obj, -1, [0, 0, 20], p.LINK_FRAME)

def steer_left(obj):
    p.applyExternalTorque(obj, -1, [0, 0, -20], p.LINK_FRAME)

def go_forward(obj):
    pass


def resetPos(obj, wall_offset):
    y_random = random.uniform(-(W/2 - wall_offset), (W/2 - wall_offset))
    yaw_random = random.uniform(-math.pi, math.pi)
    p.resetBasePositionAndOrientation(
        obj,
        [0, y_random, 0],
        p.getQuaternionFromEuler([0, 0, yaw_random])
    )
    # p.resetBasePositionAndOrientation(
    #     obj,
    #     [0, y_random, 0],
    #     p.getQuaternionFromEuler([0, 0, 0])
    # )
    p.resetBaseVelocity(obj, [0, 0, 0], [0, 0, 0])

def y_to_bin_index(y, offset_interval, y_max):
    # round y to interval of 0.25 and then convert it to its respective bin index
    y_clamped = np.clip(y, -y_max, y_max)
    normalize = y_clamped + y_max
    nearest = round(normalize / offset_interval) * offset_interval
    return int(nearest / offset_interval)

def yaw_to_bin_index(yaw, yaw_interval):
    # round y to interval of pi/6 and then give it the respective bin
    # yaw is capped -pi to pi already
    yaw_shifted = yaw + math.pi
    nearest = round(yaw_shifted / yaw_interval) * yaw_interval
    bin_index = int(nearest / yaw_interval)
    return bin_index

def state_to_bin_index(y, offset_interval, y_max, num_yaw_bins, yaw, yaw_interval):
    y_bin = y_to_bin_index(y, offset_interval, y_max)
    yaw_bin = yaw_to_bin_index(yaw, yaw_interval)
    return int(y_bin*num_yaw_bins + yaw_bin)


#reward function
def calculateReward(y, yaw, y_max=4.0, yaw_max=math.pi, stuck=False):
    if stuck or abs(y) > y_max:
        return -1
    # normalize the reward from 0 to 1
    lateral_cost = (abs(y)/y_max)**2
    yaw_cost = ((abs(yaw)-getIdealYaw(y))/yaw_max)**2
    # weights
    w_y = 1.0     # y reward param
    w_h = 0.3     # yaw reward param
    reward = - (w_y * lateral_cost + w_h * yaw_cost)
    return reward

# Q_table is the table, s is current state, a is current action, r is reward given s,a
# s_next is future state, finished is whether we're at the end of an episode
def Q_update(Q_table, s, a, r, s_next, finished, alpha=0.1, gamma=0.98):
    best_max = 0.0
    if not finished:
        best_max = float(np.max(Q_table[s_next]))
    reward_target = r + gamma * best_max
    Q_table[s, a] += alpha * (reward_target - Q_table[s, a])

def selectAction(table, s, epsilon):
    rand_gen = random.random()
    if rand_gen < epsilon:
        # choose a random action
        action = random.randrange(num_actions)
    else:
        action = int(np.argmax(table[s]))
    return action
ALPHA = 0.10
GAMMA = 0.98
EPS_MIN = 0.05
EPS_DECAY = 0.999

MAX_STEPS_PER_EP = int(6.0 / dt)   # 6 seconds for a theoretical episode
START_WALL_OFFSET = 1.25

# map action -> torque helper
def executeAction(action, obj):
    if action == 0:      # forward (no extra torque)
        go_forward(obj)
    elif action == 1:    # steer right
        steer_right(obj)
    elif action == 2:    # steer left
        steer_left(obj)

def forward_push_from_yaw(yaw, speed):
    vx = speed * math.cos(yaw)
    vy = speed * math.sin(yaw)
    return vx, vy

def run_episode(QTable, epsilon):
    resetPos(box, START_WALL_OFFSET)
    total_reward = 0.0
    done = False

    for t in range(MAX_STEPS_PER_EP):

        #get current state from offset and yaw
        (_, y, _), q = p.getBasePositionAndOrientation(box)
        yaw = getCurrAngle(q)
        s = state_to_bin_index(y, offset_interval, y_max, num_yaw_bins, yaw, yaw_interval)

        # choose an action (deendong on epsilon and state)
        a = selectAction(QTable, s, epsilon)

        #conduct action and add velocity
        executeAction(a, box)
        vx, vy = forward_push_from_yaw(yaw, speed)
        p.resetBaseVelocity(box, [vx, vy, 0])

        # simulate one step
        p.stepSimulation()
        # time.sleep(dt)

        #look at the new state and action to update table
        (_, y2, _), q2 = p.getBasePositionAndOrientation(box)
        yaw2 = getCurrAngle(q2)

        r = calculateReward(y2, yaw2, y_max=y_max, yaw_max=math.pi, stuck=False)
        done = (abs(y2) > y_max)  # out of lane bounds ends episode

        s_next = state_to_bin_index(y2, offset_interval, y_max, num_yaw_bins, yaw2, yaw_interval)

        #update table
        Q_update(QTable, s, a, r, s_next, done, alpha=ALPHA, gamma=GAMMA)

        total_reward += r
        if done:
            break

    return total_reward, t+1

#actual running code
N_EPISODES = 3000
returns = []
for ep in range(1, N_EPISODES + 1):
    G, steps_taken = run_episode(QTable, epsilon)
    returns.append(G)
    epsilon = max(EPS_MIN, EPS_DECAY * epsilon)

    if ep % 10 == 0:
        avg = sum(returns[-10:]) / min(10, len(returns))
        print(f"ep {ep:4d} | avg_return(10) = {avg: .3f} | eps={epsilon: .3f} | steps={steps_taken}")

#random matplotlib code to see graph of average reward
def moving_average(x, window=50):
    if window <= 1 or window > len(x):
        return np.array(x), np.arange(1, len(x)+1)
    ma = np.convolve(x, np.ones(window, dtype=float)/window, mode='valid')
    episodes = np.arange(window, len(x)+1)
    return ma, episodes

window = 50  # tweak to taste (e.g., 20/50/100)
ma, ma_eps = moving_average(returns, window=window)

plt.figure()
plt.plot(ma_eps, ma)
plt.xlabel("Episode")
plt.ylabel(f"Average Return (window={window})")
plt.title("Training Curve: Avg Return per Episode")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

#save data into a json file for a future greedy test
np.save("qtable.npy", QTable)  # fast + preserves shape/dtype

meta = {
    "offset_interval": offset_interval,
    "y_max": y_max,
    "yaw_interval": yaw_interval,
    "num_y_bins": num_y_bins,
    "num_yaw_bins": num_yaw_bins,
    "num_actions": num_actions
}
with open("qtable_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved policy to qtable.npy and qtable_meta.json")