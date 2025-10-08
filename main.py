import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)

p.loadURDF("plane.urdf")

p.setGravity(0, 0, -9.8)

time.sleep(5)

p.disconnect()
