import joblib
import mujoco
import mujoco_warp as mjw
import numpy as np
import os
import yaml

import torch
from scipy.spatial.transform import Rotation
from IPython.display import Video, display
from PIL import Image as PILImage
from torch.amp import autocast
# Try to import viewer (available in MuJoCo 3.0+)
try:
    import mujoco.viewer
    HAS_VIEWER = True
except (ImportError, AttributeError):
    HAS_VIEWER = False
    print("Note: mujoco.viewer not available. Interactive viewer will be disabled.")

# Try to import imageio for video saving
try:
    import imageio
    try:
        import imageio.plugins.ffmpeg
        HAS_FFMPEG = True
    except:
        HAS_FFMPEG = False
        print("Note: imageio[ffmpeg] not installed. Install with: pip install imageio[ffmpeg]")
except ImportError:
    imageio = None
    HAS_FFMPEG = False
    print("Note: imageio not available. Video saving will be disabled.")

from torch.utils._pytree import tree_map
from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel
from bfm_zero_inference_code.inference.rewards import (
    MJWArmsReward,
    MJWLocomotionReward,
    MJWMoveArmsReward,
    MJWRotationReward,
    MJWSitOnGroundReward,
    MJWSpinArmsReward,
    MJWMoveAndRotateReward,
)
# Import common configuration values
from common_23dof import ACTION_SCALES, KP_GAINS, KD_GAINS, DEFAULT_JOINT_POS, ACTION_RESCALE

MODEL_PATH = "./results/bfmzero-isaac-23dof-new/checkpoint/model"

# Get the current directory (works in both scripts and notebooks)
# In notebooks, use current working directory instead of __file__
_THIS_FILE_DIR = os.getcwd()
SAMPLE_DATA_FILE = os.path.join(_THIS_FILE_DIR, "lafan_walk_data_23dof.npz")

ROBOT_XML = str("bfm_zero_inference_code/g1_23dof_rev_1_0.xml")
QPOS_DIM = 23 + 7  # 23 dof + 7 freejoint

# Create video directory for saving all simulation videos
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Load the model
global model
model = FBcprAuxModel.load(MODEL_PATH, device="cpu")

# Sample an observation to see its structure
observation = model.obs_space.sample()

print("📊 OBSERVATION STRUCTURE")
print("=" * 60)
print(f"Type: {type(observation)}")
if isinstance(observation, dict):
    print(f"\nKeys: {list(observation.keys())}")
    print("\nComponent Details:")
    for key, value in observation.items():
        if hasattr(value, 'shape'):
            print(f"  • {key:20s}: shape={str(value.shape):15s}")
        else:
            print(f"  • {key:20s}: {type(value)}")
else:
    print(f"Shape: {observation.shape if hasattr(observation, 'shape') else 'N/A'}")

# Create a random latent z (with time-based seed for different values each run)
print("🎨 LATENT Z (Behavior Encoding)")
print("=" * 60)
print(f"Latent z is a {model.cfg.archi.z_dim}-dimensional vector that encodes desired behaviors.")
print("Let's create a random z and see how the policy uses it:\n")

import time
torch.manual_seed(int(time.time() * 1000) % 2**32)  # Use current time as seed
random_z = torch.randn(model.cfg.archi.z_dim) 
random_z = model.project_z(random_z)

print(f"Random z shape: {random_z.shape}")
print(f"Random z norm: {random_z.norm():.2f}")

# Policy inference: Takes observation + latent z → produces actions
print("🤖 Policy Output")
print("=" * 60)

def get_action_from_policy(model, obs, latent_z):
    """
    Get action from policy given observation and latent z.
    
    Args:
        obs: Observation dict
        latent_z: Latent z vector (256 dim) for the policy
    
    Returns:
        action: Action from policy (29 dim numpy array)
    """
    # Convert latent_z to tensor if needed
    if not isinstance(latent_z, torch.Tensor):
        latent_z = torch.tensor(latent_z, dtype=torch.float32)
    
    # Convert observation to torch tensors and add batch dimension
    obs_torch = tree_map(lambda x: torch.tensor(x).unsqueeze(0), obs)
    
    # Get action from policy
    with torch.no_grad():
        action = model.act(obs_torch, latent_z.unsqueeze(0), mean=True)
    
    return action[0].cpu().numpy()
    
output = get_action_from_policy(model, observation, random_z)

print(f"\nPolicy output (actions) shape: {output.shape}")

# Example: Run policy with random latent z
from env_23dof import MuJoCoBFMZeroEnv23DOF

# Initialize environment with video recording
env = MuJoCoBFMZeroEnv23DOF(
    robot_xml=ROBOT_XML,
    kp_gains=KP_GAINS,
    kd_gains=KD_GAINS,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scales=ACTION_SCALES,
    action_rescale=ACTION_RESCALE,
    enable_video=True,
    video_path="videos/random_z_simulation.mp4",
    video_fps=50  # Higher FPS for smoother playback
)

def relabel_rewards(qpos, qvel, ctrl, reward_function, batch_size=100):
    """
    Compute rewards for a batch of states using MuJoCo Warp.
    
    Args:
        qpos: Joint positions
        qvel: Joint velocities
        ctrl: Control actions
        reward_function: Reward function to apply
        batch_size: Batch size for parallel computation
    
    Returns:
        rewards: Computed rewards for each state
    """
    mjm = mujoco.MjModel.from_xml_path(ROBOT_XML)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=batch_size, nconmax=200)
    rewards = torch.zeros(qpos.shape[0], dtype=torch.float32)
    for i in range(0, qpos.shape[0], batch_size):
        mb = min(batch_size, qpos.shape[0] - i)
        if mb < batch_size:
            d = mjw.make_data(mjm, nworld=mb, nconmax=200)
        aa = reward_function(mjm, m, d, qpos[i : i + mb], qvel[i : i + mb], ctrl[i : i + mb])
        rewards[i : i + mb] = aa

    return rewards

def reward_inference_example(reward_function):
    # See load_data_from_motionlib.py for example using the motionlib .pkl files
    # Ideally you should have at least ~100k samples (more is better. Samples from LAFAN motions work best with BFM-Zero)
    sample_data = np.load(SAMPLE_DATA_FILE)
    qpos = sample_data["qpos"]
    qvel = sample_data["qvel"]
    ctrl = sample_data["action"]
    next_obs = {
        "state": torch.tensor(sample_data["next_obs.state"], dtype=torch.float32),
        "privileged_state": torch.tensor(sample_data["next_obs.privileged_state"], dtype=torch.float32),
        "last_action": torch.tensor(sample_data["next_obs.last_action"], dtype=torch.float32),
    }
    rewards = relabel_rewards(qpos, qvel, ctrl, reward_function, batch_size=100)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    z_reward = model.reward_wr_inference(next_obs, rewards)

    print("Inferred reward-weighted latent z 's shape:", z_reward.shape)
    print(f"Reward-weighted latent z norm: {torch.norm(z_reward):.2f}")
    return z_reward

reward_function = MJWMoveAndRotateReward()
reward_function.move_speed = 0.02
reward_function.target_ang_velocity = 0

z_reward = reward_inference_example(reward_function)

# Run policy with reward-inferred latent z
# Use the inferred reward z (or create one if not available)
z_to_use = z_reward[0] if len(z_reward.shape) > 1 else z_reward

# Reset environment
obs = env.reset()
env.video_path = "videos/reward_inference_simulation.mp4"
print("🚀 Running simulation with reward-inferred latent z...")
print(f"Initial root height: {env.mjd.qpos[2]:.4f}\n")

# Run simulation (fewer steps for faster video)
num_steps = 201

for step in range(num_steps):
    # Get action from policy using reward z
    action = get_action_from_policy(model, obs, z_to_use)
    
    # Step environment
    obs, next_obs, reward, info = env.step(action)
    obs = next_obs
    
    # Print progress every 50 steps
    if step % 100 == 0:
        print(f"Step {step:4d}: Root height = {info['root_height']:.4f}, Position = {info['root_position']}")

print("\n✅ Simulation complete!")

# Save video
if env.enable_video:
    env.save_video()
    print(f"🎥 Video saved to: {env.video_path}")
    
    # Display video in notebook
    print("\n📹 Video Preview:")
    display(Video(env.video_path, width=640, height=480))
