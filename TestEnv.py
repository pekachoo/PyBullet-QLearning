# print_box_pos.py
import time
import pybullet as p
import pybullet_data as pd

def main(gui=True):
    cid = p.connect(p.GUI if gui else p.DIRECT)

    try:
        p.setAdditionalSearchPath(pd.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Ground
        p.loadURDF("plane.urdf")

        # Simple box (1.0 x 1.0 x 0.3)
        half_extents = [0.5, 0.5, 0.15]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.6, 1.0, 1.0])
        box = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, 1.0],  # start 1 m above ground
        )

        print("Started. Printing box position every 0.5 s (Ctrl+C to quit).")

        # Run sim; print at 0.5 s intervals
        print_interval = 0.5
        next_print = time.time() + print_interval

        while True:
            p.applyExternalForce(box, -1, [0, 30, 0], [0, 0, 0], p.LINK_FRAME)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)  # keep pace with timestep

            now = time.time()
            if now >= next_print:
                pos, orn = p.getBasePositionAndOrientation(box)
                print(f"time={now:.2f}s  pos={pos}")
                next_print += print_interval

    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()

if __name__ == "__main__":
    main(gui=True)
