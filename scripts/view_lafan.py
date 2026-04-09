import mujoco
import mujoco.viewer
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot
import time

def replay_motion(motion_data, model_path, title="Motion Replay"):
    """Replay motion from loaded data using dof field directly."""
    first_key = list(motion_data.keys())[500]
    motion = motion_data[first_key]

    # Use dof field directly (simpler!)
    dof = motion['dof']  # (T, N) where N is 29 for 29-DOF, 23 for 23-DOF
    root_trans = motion['root_trans_offset']  # (T, 3)
    root_rot = motion['root_rot']  # (T, 4) in quaternion format
    root_rot = root_rot[:, [3, 0, 1, 2]]  # Convert from (x, y, z, w) to (w, x, y, z)
    fps = motion['fps']

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Get joint names from model
    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_names.append(name)

    print(f"{title}:")
    print(f"  dof shape: {dof.shape}")
    print(f"  root_trans shape: {root_trans.shape}")
    print(f"  root_rot shape: {root_rot.shape}")
    print(f"  Found {len(joint_names)} joints in model")
    print(f"  FPS: {fps}")

    # Replay the motion
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_idx = 0
        while viewer.is_running():
            # Set root position and orientation
            data.qpos[:3] = root_trans[t_idx]  # x, y, z position
            data.qpos[3:7] = root_rot[t_idx]    # quaternion (w, x, y, z)
            
            # Set joint angles directly from dof
            n_joints = min(len(joint_names), dof.shape[1])
            for i in range(n_joints):
                data.qpos[7 + i] = dof[t_idx, i]
            
            mujoco.mj_forward(model, data)
            
            with viewer.lock():
                viewer.sync()
            
            t_idx = (t_idx + 1) % len(dof)
            time.sleep(1.0 / fps)
        
        print(f"{title} viewer closed.")


if __name__ == "__main__":
    # View 29-DOF data
    print("=" * 50)
    print("Viewing 29-DOF data")
    print("=" * 50)
    motion_data_29dof = joblib.load("humanoidverse/data/lafan_29dof_10s-clipped.pkl")
    replay_motion(
        motion_data_29dof,
        "humanoidverse/data/robots/g1/g1_29dof.xml",
        "29-DOF"
    )

    # View 23-DOF data
    print("\n" + "=" * 50)
    print("Viewing 23-DOF data")
    print("=" * 50)
    motion_data_23dof = joblib.load("humanoidverse/data/lafan_23dof_10s-clipped.pkl")
    replay_motion(
        motion_data_23dof,
        "humanoidverse/data/robots/g1/g1_23dof_rev_1_0.xml",
        "23-DOF"
    )