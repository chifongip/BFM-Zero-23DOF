"""
Common configuration values for BFM-Zero MuJoCo simulation.

This module defines PD control gains, action scales, and default joint positions
based on BFM-Zero/config/policy/motivo_newG1.yaml
"""

import numpy as np

# Policy joint names in order (29 joints)
POLICY_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint'
]

# PD control proportional gains (Kp) for each joint (in policy_joint_names order)
KP_GAINS = np.array([
    40.1792,   # left_hip_pitch_joint (.*_hip_pitch_joint)
    99.0984,    # left_hip_roll_joint (.*_hip_roll_joint)
    40.1792,    # left_hip_yaw_joint (.*_hip_yaw_joint)
    99.0984,    # left_knee_joint (.*_knee_joint)
    28.5012,    # left_ankle_pitch_joint (.*ankle_pitch_joint)
    28.5012,    # left_ankle_roll_joint (.*ankle_roll_joint)
    40.1792,   # right_hip_pitch_joint (.*_hip_pitch_joint)
    99.0984,    # right_hip_roll_joint (.*_hip_roll_joint)
    40.1792,    # right_hip_yaw_joint (.*_hip_yaw_joint)
    99.0984,    # right_knee_joint (.*_knee_joint)
    28.5012,    # right_ankle_pitch_joint (.*ankle_pitch_joint)
    28.5012,    # right_ankle_roll_joint (.*ankle_roll_joint)
    40.1792,    # waist_yaw_joint (waist_yaw_joint)
    14.2506,    # left_shoulder_pitch_joint (.*_shoulder_.*)
    14.2506,    # left_shoulder_roll_joint (.*_shoulder_.*)
    14.2506,    # left_shoulder_yaw_joint (.*_shoulder_.*)
    14.2506,    # left_elbow_joint (.*_elbow_joint)
    14.2506,    # left_wrist_roll_joint (.*_wrist_roll_joint)
    14.2506,    # right_shoulder_pitch_joint (.*_shoulder_.*)
    14.2506,    # right_shoulder_roll_joint (.*_shoulder_.*)
    14.2506,    # right_shoulder_yaw_joint (.*_shoulder_.*)
    14.2506,    # right_elbow_joint (.*_elbow_joint)
    14.2506,    # right_wrist_roll_joint (.*_wrist_roll_joint)
], dtype=np.float32)

# PD control derivative gains (Kd) for each joint (in policy_joint_names order)
KD_GAINS = np.array([
    2.5579,    # left_hip_pitch_joint (.*_hip_pitch_joint)
    6.3088,     # left_hip_roll_joint (.*_hip_roll_joint)
    2.5579,     # left_hip_yaw_joint (.*_hip_yaw_joint)
    6.3088,     # left_knee_joint (.*_knee_joint)
    1.8145,     # left_ankle_pitch_joint (.*ankle_pitch_joint)
    1.8145,     # left_ankle_roll_joint (.*ankle_roll_joint)
    2.5579,    # right_hip_pitch_joint (.*_hip_pitch_joint)
    6.3088,     # right_hip_roll_joint (.*_hip_roll_joint)
    2.5579,     # right_hip_yaw_joint (.*_hip_yaw_joint)
    6.3088,     # right_knee_joint (.*_knee_joint)
    1.8145,     # right_ankle_pitch_joint (.*ankle_pitch_joint)
    1.8145,     # right_ankle_roll_joint (.*ankle_roll_joint)
    2.5579,     # waist_yaw_joint (waist_yaw_joint)
    0.9072,     # left_shoulder_pitch_joint (.*_shoulder_.*)
    0.9072,     # left_shoulder_roll_joint (.*_shoulder_.*)
    0.9072,     # left_shoulder_yaw_joint (.*_shoulder_.*)
    0.9072,     # left_elbow_joint (.*_elbow_joint)
    0.9072,    # left_wrist_roll_joint (.*_wrist_roll_joint)
    0.9072,     # right_shoulder_pitch_joint (.*_shoulder_.*)
    0.9072,     # right_shoulder_roll_joint (.*_shoulder_.*)
    0.9072,     # right_shoulder_yaw_joint (.*_shoulder_.*)
    0.9072,     # right_elbow_joint (.*_elbow_joint)
    0.9072,    # right_wrist_roll_joint (.*_wrist_roll_joint)
], dtype=np.float32)

# Default joint positions for each joint (in policy_joint_names order)
# Only hip_pitch, knee, and ankle_pitch have non-zero defaults
DEFAULT_JOINT_POS = np.array([
    -0.1,   # left_hip_pitch_joint (.*_hip_pitch_joint)
    0.0,    # left_hip_roll_joint
    0.0,    # left_hip_yaw_joint
    0.3,    # left_knee_joint (.*_knee_joint)
    -0.2,   # left_ankle_pitch_joint (.*_ankle_pitch_joint)
    0.0,    # left_ankle_roll_joint
    -0.1,   # right_hip_pitch_joint (.*_hip_pitch_joint)
    0.0,    # right_hip_roll_joint
    0.0,    # right_hip_yaw_joint
    0.3,    # right_knee_joint (.*_knee_joint)
    -0.2,   # right_ankle_pitch_joint (.*_ankle_pitch_joint)
    0.0,    # right_ankle_roll_joint
    0.0,    # waist_yaw_joint
    0.0,    # left_shoulder_pitch_joint
    0.0,    # left_shoulder_roll_joint
    0.0,    # left_shoulder_yaw_joint
    0.0,    # left_elbow_joint
    0.0,    # left_wrist_roll_joint
    0.0,    # right_shoulder_pitch_joint
    0.0,    # right_shoulder_roll_joint
    0.0,    # right_shoulder_yaw_joint
    0.0,    # right_elbow_joint
    0.0,    # right_wrist_roll_joint
], dtype=np.float32)

DOF_EFFORT_LIMITS = np.array([
    88.0,
    139.0,
    88.0,
    139.0,
    35.0,
    35.0,
    88.0,
    139.0,
    88.0,
    139.0,
    35.0,
    35.0,
    88.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
], dtype=np.float32)

# Global action rescale factor
ACTION_RESCALE = 5.0

ACTION_SCALES = 0.25 * DOF_EFFORT_LIMITS / KP_GAINS
