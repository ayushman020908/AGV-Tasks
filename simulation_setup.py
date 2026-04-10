"""
simulation_setup.py

Isolated setup module for setting up the simulation environment required for the given task.

Contains:
  - make_obstacle_texture()   — yellow/black checkerboard texture
  - create_road_and_obstacles() — road surface, lane markings, slalom obstacles, end wall
  - create_car()              — loads and configures the racecar URDF
  - setup_simulation()        — convenience wrapper that initialises everything

Usage (standalone demo):
    python simulation_setup.py

Usage (as a module):
    from simulation_setup import setup_simulation
    carId, steering_joints, motor_joints = setup_simulation()
"""

import cv2
import numpy as np
import tempfile
import pathlib
import pybullet as p
import pybullet_data
import time


def make_obstacle_texture(size: int = 128, tile: int = 10) -> str:
    """
    Creates a yellow (#FFD700) / black checkerboard PNG and returns its path.
    Parameters:
    size : int: Image size in pixels (square).     
    tile : int: Checkerboard tile size in pixels.       

    Returns:
    str : Absolute path to the saved PNG texture file.
    """
    img    = np.zeros((size, size, 3), dtype=np.uint8)
    black  = np.array([0,   0,   0],   dtype=np.uint8)  # BGR black
    yellow = np.array([0,   215, 255], dtype=np.uint8)  # BGR yellow (#FFD700)

    for row in range(size):
        for col in range(size):
            img[row, col] = black if (row // tile + col // tile) % 2 == 0 else yellow

    tex_path = pathlib.Path(tempfile.gettempdir()) / "obs_texture.png"
    ok = cv2.imwrite(str(tex_path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed writing texture to: {tex_path}")

    print(f"[Texture] Saved yellow/black checkerboard → {tex_path}")
    return str(tex_path)


# =============================================================================
# ROAD + OBSTACLES
# =============================================================================

def create_road_and_obstacles():
    """
    Builds the full scene in the currently connected PyBullet world:

      - Road surface     : dark grey box, 33.3 m long × 2.32 m wide
      - Lane markings    : dense white dashes at y = 0, ±0.85 m
      - Slalom obstacles : 5 yellow/black textured boxes in alternating lateral positions
      - End wall         : blue box at x ≈ 31.7 m

    Road half-width is 1.16 m on each side of centre (local frame).
    Obstacle positions (x = 6, 12, 18, 24 m) alternate y = +0.38 / −0.38 m.

    Must be called after p.connect() and p.loadURDF("plane.urdf").
    """
    tex_path = make_obstacle_texture(size=128, tile=10)
    tex_id   = p.loadTexture(tex_path)

    # ── Road surface ──────────────────────────────────────────────────────
    rv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[16.66, 1.16, 0.01],
        rgbaColor=[0.15, 0.15, 0.15, 1]
    )
    rc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[16.66, 1.16, 0.01])
    p.createMultiBody(0, rc, rv, [16.66, 0, 0.01])

    # ── Lane markings — dense white dashes for background optical flow ────
    for x in np.arange(0, 34, 0.4):
        lv = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.12, 0.03, 0.01],
            rgbaColor=[1, 1, 1, 1]
        )
        p.createMultiBody(0, -1, lv, [x,  0.0,  0.02])
        p.createMultiBody(0, -1, lv, [x,  0.85, 0.02])
        p.createMultiBody(0, -1, lv, [x, -0.85, 0.02])

    # ── Slalom obstacles — white base so yellow/black texture shows fully ─
    obs_extents = [0.25, 0.45, 0.35]
    for i, x in enumerate(range(6, 30, 6)):
        y      = 0.38 if i % 2 == 0 else -0.38
        ov     = p.createVisualShape(
            p.GEOM_BOX, halfExtents=obs_extents,
            rgbaColor=[1, 1, 1, 1]
        )
        oc     = p.createCollisionShape(p.GEOM_BOX, halfExtents=obs_extents)
        obs_id = p.createMultiBody(10, oc, ov, [x, y, 0.35])
        p.changeVisualShape(obs_id, -1, textureUniqueId=tex_id)
        print(f"[Obstacle {i}] x={x}m  y={y:+.2f}m  texture applied")

    # ── End wall (blue — visually distinct from obstacles) ────────────────
    wv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.5, 1.16, 0.5],
        rgbaColor=[0.1, 0.3, 0.9, 1]
    )
    wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 1.16, 0.5])
    p.createMultiBody(0, wc, wv, [31.66, 0, 0.5])
    print("[End wall] placed at x=31.66 m")


# =============================================================================
# CAR
# =============================================================================

def create_car(start_pos=None, start_orn=None, global_scaling=1.8):
    """
    Loads the PyBullet racecar URDF and returns its ID along with
    categorised joint lists

    Parameters:
    start_pos : list[float], optional: [x, y, z] spawn position. Defaults to [0, 0, 0.25].
    start_orn : list[float], 
    global_scaling : float: URDF scaling factor (1.8 matches the road width).

    Returns:
    car_id : int: PyBullet body ID of the car.
    steering_joints : list[int]: Joint indices whose names contain 'steer'.
    motor_joints : list[int]: Joint indices whose names contain 'wheel'.
    """
    if start_pos is None:
        start_pos = [0, 0, 0.25]
    if start_orn is None:
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

    car_id = p.loadURDF(
        "racecar/racecar.urdf",
        start_pos, start_orn,
        globalScaling=global_scaling
    )
    p.changeDynamics(car_id, -1, ccdSweptSphereRadius=0.1)

    steering_joints, motor_joints = [], []
    for i in range(p.getNumJoints(car_id)):
        name = p.getJointInfo(car_id, i)[1].decode('utf-8')
        if 'steer' in name.lower():
            steering_joints.append(i)
        elif 'wheel' in name.lower():
            motor_joints.append(i)

    print(f"[Car] body_id={car_id}  "
          f"steering_joints={steering_joints}  "
          f"motor_joints={motor_joints}")
    return car_id, steering_joints, motor_joints


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def setup_simulation(dt=1.0 / 60.0, settle_frames=60, gui=True):
    """
    Full simulation initialisation in one call.

    Steps
    -----
    1. Connect to PyBullet (GUI or DIRECT).
    2. Set gravity and load the ground plane.
    3. Build road, lane markings, obstacles, and end wall.
    4. Spawn the racecar and settle its suspension.

    Parameters
    ----------
    dt : float: Physics timestep in seconds. Default 1/60 s.
    settle_frames : int: Number of physics steps to run before returning, so the car
        suspension reaches equilibrium. Default 60.
    gui : bool : If True, opens the PyBullet GUI window. If False, runs headless
        (DIRECT mode) — useful for unit tests or batch runs.

    Returns
    -------
    car_id : int
    steering_joints : list[int]
    motor_joints : list[int]
    """
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    p.loadURDF("plane.urdf")

    create_road_and_obstacles()
    car_id, steering_joints, motor_joints = create_car()

    print(f"[Setup] Settling suspension for {settle_frames} frames …")
    for _ in range(settle_frames):
        p.stepSimulation()
        time.sleep(dt)
    print("[Setup] Ready.")

    return car_id, steering_joints, motor_joints


# =============================================================================
# STANDALONE DEMO
# =============================================================================

if __name__ == "__main__":
    car_id, steer_j, motor_j = setup_simulation()

    print("\nSimulation running. Close the PyBullet window or press Ctrl+C to exit.")
    dt = 1.0 / 60.0
    try:
        while True:
            # Spin the wheels at a gentle speed so you can see the car move
            for j in motor_j:
                p.setJointMotorControl2(
                    car_id, j,
                    p.VELOCITY_CONTROL,
                    targetVelocity=5.0,
                    force=800
                )
            p.stepSimulation()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            p.disconnect()
        except Exception:
            pass
        print("Simulation ended.")