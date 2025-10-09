import pybullet as p, pybullet_data as pd, time, math

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
    rollingFriction=0.005,  # lowkey irrelevant
    spinningFriction=0.01   # slight resistance to yaw movement
)

dt = 1/240
force = [10, 0, 0]  #newtons
application_point = [0,0.7,0]  # at center
steps = int(10.0/dt)  # 10 seconds total with dt steps in between each second

force2 = [10, 0, 0]
application_point2 = [0,-0.5,0]

for i in range(steps):
    p.applyExternalForce(box, -1, forceObj=force, posObj=application_point, flags=p.WORLD_FRAME)
    p.applyExternalForce(box, -1, forceObj=force2, posObj=application_point2, flags=p.WORLD_FRAME)
    p.stepSimulation()
    time.sleep(dt)
