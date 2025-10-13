import pybullet as p, pybullet_data as pd, time, math, random

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

W = 10.0  # width
L = 100.0
wall_th = 0.1
wall_h = 1.0
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
    rollingFriction=0.05,  # lowkey irrelevant
    spinningFriction=0.15   # resist rotation
)
magnitude = 15

left_force = magnitude
right_force = 5

increment = magnitude/100
forward = True

dt = 1/240
steps = int(100.0/dt)  # 10 seconds total with dt steps in between each second

def getCurrAngle(quaternion):
    _, _, yaw = p.getEulerFromQuaternion(quaternion)
    return yaw

def PDCalculation(error, derivative, kp=65, kd=0.3):
    return kp*error + kd*derivative

def getIdealYaw(offset, lookahead=5):
    #cap it out at pi/2 and -pi/2
    if offset > lookahead:
        offset = lookahead
    elif offset < -lookahead:
        offset = -lookahead
    return -math.atan2(offset, math.sqrt(math.pow(lookahead, 2) - math.pow(offset, 2)))



error = 0
lastError = 0
speed = 2.0

message_interval = 5.0
next_interval = message_interval
current_time = 0

def resetPos(obj, wall_offset):
    y_random = random.uniform(-(W/2 - wall_offset), (W/2 - wall_offset))
    yaw_random = random.uniform(-math.pi, math.pi)
    p.resetBasePositionAndOrientation(
        obj,
        [0, y_random, 0],
        p.getQuaternionFromEuler([0, 0, yaw_random])
    )
    p.resetBaseVelocity(obj, [0, 0, 0], [0, 0, 0])

for i in range(steps):
    (x, y, _), q = p.getBasePositionAndOrientation(box)
    yaw = getCurrAngle(q)
    error = getIdealYaw(y) - yaw
    # derivative = (error - lastError)/dt
    derivative = 0
    # p.applyExternalTorque(box, -1, [0, 0, PDCalculation(error, derivative)], p.LINK_FRAME)
    p.applyExternalTorque(box, -1, [0, 0, 10], p.LINK_FRAME)
    vx = speed * math.cos(yaw)
    vy = speed * math.sin(yaw)

    #send a forward velocity
    p.resetBaseVelocity(box, [vx, vy, 0])
    print(x)
    # print(y)
    # print(error)

    if current_time > next_interval:
        resetPos(box, 1)
        next_interval += message_interval

    p.stepSimulation()
    time.sleep(dt)
    lastError = error
    current_time += dt

