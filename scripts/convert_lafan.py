import joblib
import numpy as np

def convert_29dof_to_23dof(input_pkl_path, output_pkl_path):
    """
    Convert 29-DOF motion data to 23-DOF format.
    
    Joints removed (6 total):
    - waist_roll_joint (29-DOF index 13)
    - waist_pitch_joint (29-DOF index 14)
    - left_wrist_pitch_joint (29-DOF index 20)
    - left_wrist_yaw_joint (29-DOF index 21)
    - right_wrist_pitch_joint (29-DOF index 27)
    - right_wrist_yaw_joint (29-DOF index 28)
    """
    
    # Load 29-DOF data
    print(f"Loading 29-DOF data from {input_pkl_path}.")
    motion_data_29dof = joblib.load(input_pkl_path)
    
    # Indices to keep from 29-DOF dof array to get 23-DOF
    dof_indices_to_keep = [
        # Left leg (6)
        0, 1, 2, 3, 4, 5,
        # Right leg (6)
        6, 7, 8, 9, 10, 11,
        # Waist (1) - only yaw
        12,
        # Left arm (5) - only pitch, roll, yaw, elbow, wrist_roll
        15, 16, 17, 18, 19,
        # Right arm (5) - only pitch, roll, yaw, elbow, wrist_roll
        22, 23, 24, 25, 26
    ]
    
    # Convert each motion
    converted_data = {}
    for motion_key, motion_data in motion_data_29dof.items():
        print(f"Converting: {motion_key}")
        
        # Get original data
        root_trans = motion_data['root_trans_offset']  # (T, 3)
        pose_aa_29dof = motion_data['pose_aa']  # (T, 30, 3)
        fps = motion_data['fps']
        
        # Extract root rotation (first joint, index 0)
        root_rot = pose_aa_29dof[:, 0, :]  # (T, 3)
        
        # Extract DOF from pose_aa (joints 1-29, which are indices 1-29 in pose_aa)
        # pose_aa[:, 1:, :] gives (T, 29, 3) for the 29 child joints
        dof_29dof = pose_aa_29dof[:, 1:, :]  # (T, 29, 3)
        
        # Select only the joints we want to keep
        dof_23dof = dof_29dof[:, dof_indices_to_keep, :]  # (T, 23, 3)
        
        # Reconstruct pose_aa for 23-DOF
        # Root joint at index 0, then 23 child joints
        pose_aa_23dof = np.concatenate([root_rot[:, None, :], dof_23dof], axis=1)  # (T, 24, 3)
        
        # Convert dof array (29 -> 23)
        if 'dof' in motion_data:
            dof_29 = motion_data['dof']  # (T, 29)
            dof_23 = dof_29[:, dof_indices_to_keep]  # (T, 23)
        else:
            dof_23 = None
        
        # Create new motion data dict
        converted_data[motion_key] = {
            'root_trans_offset': root_trans.astype(motion_data['root_trans_offset'].dtype),
            'pose_aa': pose_aa_23dof.astype(motion_data['pose_aa'].dtype),
            'fps': int(fps),
            'motion_name': motion_data.get('motion_name', motion_key),
        }
        
        # Add dof if it exists
        if dof_23 is not None:
            converted_data[motion_key]['dof'] = dof_23.astype(motion_data['dof'].dtype)
        
        # Copy other fields if they exist
        for key in ['root_rot', 'smpl_joints']:
            if key in motion_data:
                converted_data[motion_key][key] = motion_data[key]
    
    # Save converted data
    joblib.dump(converted_data, output_pkl_path)
    print(f"Saved {len(converted_data)} motions to {output_pkl_path}")
    
    # Print verification
    first_key = list(converted_data.keys())[0]
    first_motion = converted_data[first_key]
    print(f"\nVerification:")
    print(f"  pose_aa shape: {first_motion['pose_aa'].shape} (expected: (T, 24, 3))")
    if 'dof' in first_motion:
        print(f"  dof shape: {first_motion['dof'].shape} (expected: (T, 23))")
    print(f"  root_trans_offset shape: {first_motion['root_trans_offset'].shape}")
    print(f"  fps: {first_motion['fps']}")
    
    return converted_data


# Usage
if __name__ == "__main__":
    input_path = "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
    output_path = "humanoidverse/data/lafan_23dof_10s-clipped.pkl"
    
    converted_data = convert_29dof_to_23dof(input_path, output_path)