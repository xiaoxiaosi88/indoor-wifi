# config.py
"""
Configuration parameters for the Indoor Localization DQN project.
"""

# --- Basic Configuration ---
N_APS = 520                   # Number of Access Points from CSV header (MUST MATCH DATA)
ACTION_DIM = 4                # Action space dimension (NW, NE, SW, SE)
HISTORY_LEN = 10              # Action history sequence length

#--- 计算依赖变量 ---
HISTORY_VEC_SIZE = ACTION_DIM * HISTORY_LEN # 现在 ACTION_DIM 和 HISTORY_LEN 已定义
STATE_INPUT_DIM = N_APS +3 + HISTORY_VEC_SIZE # 现在 N_APS, HISTORY_VEC_SIZE 已定义

# --- Environment & Target Parameters ---
TARGET_BUILDING_ID = 1        # Target Building ID to filter/train on
TARGET_FLOOR_ID = 2           # Target Floor ID to filter/train on
RADGT = 0.5                   # Target localization radius (meters) for ground truth window
DELTA_IOU = 0.5               # Intersection over Window (IoW) success threshold
ETA_REWARD = 3.0              # Terminal reward magnitude (+ for success, - for failure)
TAU_REWARD = 1.0              # Intermediate reward for improving IoW
MAX_STEPS_PER_EPISODE = 15    # Max steps agent takes per episode

# --- Training Hyperparameters ---
NUM_EPISODES = 1000000          # Total number of training episodes
TARGET_UPDATE_FREQ = 1000     # Frequency (in training steps/replay calls) to update the target network
BUFFER_SIZE = 100000          # Experience Replay buffer size
BATCH_SIZE = 128              # Training batch size
LEARNING_RATE = 0.0001        # Learning rate for the Adam optimizer
GAMMA = 0.1                   # Discount factor for future rewards
EPSILON_START = 1.0           # Initial exploration rate (epsilon)
EPSILON_DECAY_RATE = 0.999995 # Multiplicative decay rate per training step (replay call)
EPSILON_MIN = 0.05            # Minimum exploration rate

# --- Other Configurations ---
MODEL_SAVE_FREQ = 2000        # Frequency (in episodes) to save the model weights
# Filenames will be formatted in main.py using TARGET_BUILDING_ID and TARGET_FLOOR_ID
MODEL_FILENAME_TEMPLATE = "dqn_b{building}f{floor}.weights.h5"
PLOT_FILENAME_TEMPLATE = "dqn_b{building}f{floor}_training.png"
LOG_FILENAME_TEMPLATE = "dqn_b{building}f{floor}_training_log.csv"
DATA_CSV_FILENAME = "TrainingData1.csv"  # Source data file name
NO_SIGNAL_RSSI = -110.0       # Value to represent no signal detected
MIN_DETECTED_RSSI = -100.0    # Minimum RSSI value considered as 'detected' for stats calculation