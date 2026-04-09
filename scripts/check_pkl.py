import joblib
import torch

# Load the file
pkl_path = "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
motion_data = joblib.load(pkl_path)

print(f"Total motions: {len(motion_data)}")
print(f"Type: {type(motion_data)}")

# Check first motion
first_key = list(motion_data.keys())[0]
first_motion = motion_data[first_key]

print(f"\nFirst motion: {first_key}")
print(f"Keys: {list(first_motion.keys())}")

# Check shapes
for key, value in first_motion.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape} {value.dtype}")
    elif isinstance(value, (int, float, str)):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {type(value)}")
