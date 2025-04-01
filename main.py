# main.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import traceback
import tensorflow as tf
from datetime import datetime # --- TB: For log directory timestamp ---

# Import custom modules
from environment import SingleFloorLocalizationEnv
from agent import DQNAgent
from data_utils import load_and_process_data
# Import config
import config

def check_gpu():
    # (Keep your existing check_gpu function)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"\n--- Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs ---\n")
            print(tf.config.list_physical_devices())
        except RuntimeError as e:
            print("Error setting memory growth:", e)
    else:
        print("\n--- No GPU found, using CPU. ---\n")

def main():
    print("Starting Indoor Localization DQN Training...")
    check_gpu()

    # Load parameters from config
    N_APS = config.N_APS
    ACTION_DIM = config.ACTION_DIM
    HISTORY_LEN = config.HISTORY_LEN
    # STATE_INPUT_DIM will be recalculated based on data, but keep config for reference
    # STATE_INPUT_DIM = config.STATE_INPUT_DIM

    TARGET_BUILDING_ID = config.TARGET_BUILDING_ID
    TARGET_FLOOR_ID = config.TARGET_FLOOR_ID
    RADGT = config.RADGT
    DELTA_IOU = config.DELTA_IOU
    MAX_STEPS_PER_EPISODE = config.MAX_STEPS_PER_EPISODE##每个训练回合的最大步数

    NUM_EPISODES = config.NUM_EPISODES##总的episode
    TARGET_UPDATE_FREQ = config.TARGET_UPDATE_FREQ
    BUFFER_SIZE = config.BUFFER_SIZE
    BATCH_SIZE = config.BATCH_SIZE
    LEARNING_RATE = config.LEARNING_RATE
    GAMMA = config.GAMMA
    EPSILON_START = config.EPSILON_START
    EPSILON_DECAY_RATE = config.EPSILON_DECAY_RATE
    EPSILON_MIN = config.EPSILON_MIN

    MODEL_SAVE_FREQ = config.MODEL_SAVE_FREQ
    # Format filenames
    MODEL_FILENAME = config.MODEL_FILENAME_TEMPLATE.format(building=TARGET_BUILDING_ID, floor=TARGET_FLOOR_ID)
    PLOT_FILENAME = config.PLOT_FILENAME_TEMPLATE.format(building=TARGET_BUILDING_ID, floor=TARGET_FLOOR_ID)
    LOG_FILENAME = config.LOG_FILENAME_TEMPLATE.format(building=TARGET_BUILDING_ID, floor=TARGET_FLOOR_ID)
    DATA_CSV_FILENAME = config.DATA_CSV_FILENAME
    NO_SIGNAL_RSSI = config.NO_SIGNAL_RSSI
    MIN_DETECTED_RSSI = config.MIN_DETECTED_RSSI

    # --- TB: Create unique log directory ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = os.path.join("logs", f"dqn_b{TARGET_BUILDING_ID}f{TARGET_FLOOR_ID}_{timestamp}")
    print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    # --------------------------------------

    # --- Load and process data ---
    print(f"--- Loading data for B{TARGET_BUILDING_ID} F{TARGET_FLOOR_ID} ---")
    start_time = time.time()
    rssi_train, location_train, map_bounds_train, norm_params = load_and_process_data(
        filename=DATA_CSV_FILENAME,
        building_id=TARGET_BUILDING_ID,
        floor_id=TARGET_FLOOR_ID,
        radgt=RADGT,
        no_signal_value=NO_SIGNAL_RSSI,
        rssi_min_value=MIN_DETECTED_RSSI
    )

    if rssi_train is None:
        print("Exiting due to data loading error.")
        return

    num_aps_actual = rssi_train.shape[1]
    if num_aps_actual != N_APS:
        print(f"WARNING: Loaded data has {num_aps_actual} WAP columns, config expected {N_APS}. Using actual: {num_aps_actual}")
        N_APS = num_aps_actual # Update N_APS based on actual data

    # Recalculate STATE_INPUT_DIM based on actual data dimensions
    STATE_INPUT_DIM = num_aps_actual + 3 + (ACTION_DIM * HISTORY_LEN)##状态维度
    print(f"Recalculated STATE_INPUT_DIM: {STATE_INPUT_DIM}")

    num_training_samples = rssi_train.shape[0]##总的样本量
    print(f"Data loading & processing complete ({num_training_samples} samples) in {time.time() - start_time:.2f}s")

    # --- Initialize Environment and Agent ---
    print("Initializing Environment and Agent...")
    env = SingleFloorLocalizationEnv(
        radgt=RADGT, delta=DELTA_IOU, eta=config.ETA_REWARD, tau=config.TAU_REWARD,
        max_steps=MAX_STEPS_PER_EPISODE, history_len=HISTORY_LEN, action_dim=ACTION_DIM
    )

    agent = DQNAgent(
        state_input_dim=STATE_INPUT_DIM, action_dim=ACTION_DIM, history_len=HISTORY_LEN,
        norm_params=norm_params, # Pass calculated norm_params
        learning_rate=LEARNING_RATE, gamma=GAMMA,
        epsilon=EPSILON_START, epsilon_decay=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN,
        buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        log_dir=tensorboard_log_dir # --- TB: Pass the log directory ---
    )

    # --- Load previous model if exists ---
    if os.path.exists(MODEL_FILENAME):
        agent.load(MODEL_FILENAME)

    # --- Print final configuration confirmation ---
    print(f"\n--- Starting Training ---")
    print(f" Building: {TARGET_BUILDING_ID}, Floor: {TARGET_FLOOR_ID}")
    print(f" Episodes: {NUM_EPISODES}, Max Steps/Ep: {MAX_STEPS_PER_EPISODE}")
    print(f" Data Samples: {num_training_samples}")
    print(f" State Input Dim: {STATE_INPUT_DIM} (APs:{N_APS}+Win:3+Hist:{ACTION_DIM * HISTORY_LEN})")
    print(f" Action Dim: {ACTION_DIM}")
    print(f" Learning Rate: {LEARNING_RATE}, Gamma: {GAMMA}")
    print(f" Epsilon Start: {EPSILON_START:.4f}, Decay: {EPSILON_DECAY_RATE}, Min: {EPSILON_MIN:.4f}")
    print(f" Buffer Size: {BUFFER_SIZE}, Batch Size: {BATCH_SIZE}")
    print(f" Target Update Freq (learn steps): {TARGET_UPDATE_FREQ}")
    print(f" Save Freq (episodes): {MODEL_SAVE_FREQ}")
    print(f" Model Filename: {MODEL_FILENAME}")
    print(f" Log Filename: {LOG_FILENAME}")
    print(f" TensorBoard Log Dir: {tensorboard_log_dir}") # --- TB: Print log dir ---
    print(f" Map Bounds: {map_bounds_train}")
    print(f" Normalization Params: {norm_params}") # Print norm params used
    print("-" * 40)


    # --- Training Log and Stats ---
    # (Keep your log file setup)
    try:
        log_file = open(LOG_FILENAME, 'w')
        log_file.write("Episode,Steps,TotalReward,Success,EpsilonEnd,AvgLossEp,FinalIoW\n") # Adjusted headers slightly
    except IOError as e:
        print(f"Error opening log file {LOG_FILENAME}: {e}. Exiting.")
        return

    # Keep track of metrics for plotting and console output
    episode_rewards = []#存储每个回合的累计奖励，用于评估智能体在回合中的表现
    # average_losses = [] # Logged per step now, less critical per episode
    success_flags = []#存储每个回合的成功标志，用于判断智能体是否完成了任务
    step_counts = [] # 存储每个回合的步数，用于分析智能体完成任务所需的步数。
    final_iows = [] # 存储每个回合的最终IoU（Intersection over Union）值，用于评估智能体在任务中的表现
    # Store losses per episode for CSV log only if needed
    episode_avg_losses_for_log = []
    # 如果需要，存储每个回合的平均损失值，仅用于CSV日志记录


    # --- Training Loop ---
    training_start_time = time.time()##返回当前时间戳
    print("Beginning training loop...\n")
    try: # Wrap training loop in try...finally to ensure log file closure
        '''
        下面这一段有歧义
        '''
        for episode in range(NUM_EPISODES):##每次定位RSSI值是不变的
            ep_start_time = time.time()
            # 计算当前训练样本的索引
            # 通过取模运算确保索引在有效范围内，避免超出样本总数
            # episode: 当前训练的轮次或步数
            # num_training_samples: 训练样本的总数
            # 返回值: 当前轮次对应的训练样本索引
            data_idx = episode % num_training_samples
            state = env.reset(
                rssi_vector=rssi_train[data_idx],##每一行的所有列
                ground_truth_loc=tuple(location_train[data_idx]),##每一行的经度和维度
                bounds=map_bounds_train
            )

            if state is None:

                 print(f"\n\033[1;31m{'!'*50}\nWARNING: Episode {episode + 1} SKIPPED due to env reset failure!\n{'!'*50}\033[0m\n")
                 # Log dummy values for consistency
                 episode_rewards.append(-env.eta * MAX_STEPS_PER_EPISODE) # Estimate max penalty
                 success_flags.append(0)
                 step_counts.append(0)
                 final_iows.append(0.0)
                 episode_avg_losses_for_log.append(np.nan)
                 log_line = f"{episode + 1},0,{-env.eta * MAX_STEPS_PER_EPISODE:.2f},0,{agent.epsilon:.5f},NaN,0.0\n"
                 try:
                     log_file.write(log_line)
                     log_file.flush()
                 except Exception as log_e:
                     print(f"Error writing skipped episode to log file: {log_e}")
                 # --- TB: Log episode failure ---
                 agent.log_metrics({
                     'episode/reward': -env.eta * MAX_STEPS_PER_EPISODE,
                     'episode/steps': 0,
                     'episode/success': 0.0,
                     'episode/final_iow': 0.0,
                     'episode/avg_loss': np.nan, # Log NaN for loss
                     'episode/epsilon_end': agent.epsilon # Log epsilon at skip time
                 }, step=episode + 1)
                 # -----------------------------
                 continue

            total_reward = 0.0
            done = False
            step = 0
            episode_loss_sum = 0.0###计算当前回合所有学习步骤的损失总和，评估DQN的收敛情况
            num_losses_in_episode = 0##当前训练回合中有效损失值的数量
            is_success = 0##定位是否成功
            last_iow = 0.0##记录最后一步的IoW值，用于评估智能体的定位精度

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                step += 1
                total_reward += reward
                last_iow = info.get('iow', 0.0)

                if next_state is None:
                    print(f"Error: Env step returned None in Ep {episode+1}, Step {step}. Ending episode with penalty.")
                    reward = -env.eta # Apply penalty
                    total_reward += reward # Add penalty to total
                    done = True # Force end episode
                    # Agent cannot remember or learn from this invalid transition
                else:
                    # Store experience
                    ##replay——buffer
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state # Update state *after* remembering the old one

                    # Learn from experience (replay)
                    loss = agent.replay() # Replay happens *after* moving to next state
                    ###这段代码用于处理和累积深度Q网络(DQN)训练过程中的损失值，同时确保只记录有效的损失值。
                    if loss is not None and not np.isnan(loss):
                        episode_loss_sum += loss
                        num_losses_in_episode += 1

                # Check success based on reward signal from environment
                if reward == env.eta and done:
                    is_success = 1

                # Safety break (optional, env should handle max_steps)
                # if step >= MAX_STEPS_PER_EPISODE + 5: ...
            ##后面不是主要循环内容，上面才是
            # --- End of Episode Stats & Logging ---
            ep_duration = time.time() - ep_start_time
            avg_loss_ep = episode_loss_sum / num_losses_in_episode if num_losses_in_episode > 0 else np.nan
            episode_rewards.append(total_reward)
            success_flags.append(is_success)
            step_counts.append(step)
            episode_avg_losses_for_log.append(avg_loss_ep) # Store for CSV
            final_iows.append(last_iow)

            # Log to CSV file
            avg_loss_str = f"{avg_loss_ep:.5f}" if not np.isnan(avg_loss_ep) else "NaN"
            log_line = f"{episode+1},{step},{total_reward:.2f},{is_success},{agent.epsilon:.5f},{avg_loss_str},{last_iow:.4f}\n"
            try:
                 log_file.write(log_line)
                 log_file.flush()
            except Exception as log_e:
                 print(f"Error writing episode {episode+1} to log file: {log_e}")

            # --- TB: Log episode metrics ---
            agent.log_metrics({
                'episode/reward': total_reward,
                'episode/steps': step,
                'episode/success': float(is_success), # Ensure float
                'episode/final_iow': last_iow,
                'episode/avg_loss': avg_loss_ep, # Can be NaN
                'episode/epsilon_end': agent.epsilon # Epsilon at end of episode
            }, step=episode + 1) # Log against episode number
            # -----------------------------


            # Print progress periodically (keep this)
            if (episode + 1) % 100 == 0 or episode == NUM_EPISODES - 1:
                moving_avg_window = 100
                # Use lists populated during the loop for moving averages
                avg_rew = np.mean(episode_rewards[-moving_avg_window:])
                avg_steps = np.mean(step_counts[-moving_avg_window:])
                avg_succ = np.mean(success_flags[-moving_avg_window:]) * 100
                # Loss is logged per step, no easy moving avg here, rely on TensorBoard
                # avg_loss_recent = np.nanmean(episode_avg_losses_for_log[-moving_avg_window:])
                # loss_recent_str = f"{avg_loss_recent:.5f}" if not np.isnan(avg_loss_recent) else "NaN"

                print(f"Ep:{episode+1:>6}/{NUM_EPISODES} | Steps:{avg_steps:5.1f} | MA Rew:{avg_rew:+7.2f} | "
                      # f"MA Succ:{avg_succ:6.2f}% | Epsilon:{agent.epsilon:.4f} | MA Ep Loss:{loss_recent_str} | Ep Time:{ep_duration:.2f}s")
                      f"MA Succ:{avg_succ:6.2f}% | Epsilon:{agent.epsilon:.4f} | Ep Time:{ep_duration:.2f}s") # Removed MA Ep Loss from console

            # Save model weights periodically
            if (episode + 1) % MODEL_SAVE_FREQ == 0:
                agent.save(MODEL_FILENAME) # Agent logs save event to TB
                print(f"--- Model weights saved to {MODEL_FILENAME} at episode {episode+1} ---")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training loop: {e}")
        traceback.print_exc()
    finally:
        # --- End of Training ---
        if log_file and not log_file.closed:
             log_file.close()
             print("Log file closed.")
        total_training_time = time.time() - training_start_time
        print("-" * 40)
        print(f"Training finished or interrupted in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s.")
        agent.save(MODEL_FILENAME) # Save final model state
        print(f"Final model saved to {MODEL_FILENAME}")

    # --- Plotting Results (keep your existing plotting code) ---
    try:
        if os.path.exists(LOG_FILENAME):
             log_data = pd.read_csv(LOG_FILENAME)
             plot_window_size = 200 # For smoothing plots

             fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

             # Reward Plot
             axs[0].plot(log_data['Episode'], log_data['TotalReward'], label='Episode Reward', alpha=0.3, linewidth=0.8)
             axs[0].plot(log_data['Episode'], log_data['TotalReward'].rolling(window=plot_window_size, min_periods=1).mean(),
                         label=f'Mov Avg Reward ({plot_window_size} eps)', color='red')
             axs[0].set_ylabel("Total Reward")
             axs[0].set_title(f"Training Rewards (B{TARGET_BUILDING_ID} F{TARGET_FLOOR_ID})")
             axs[0].legend(loc='upper left')
             axs[0].grid(True)

             # Success Rate Plot (on same axes as reward)
             ax0_twin = axs[0].twinx()
             ax0_twin.plot(log_data['Episode'], log_data['Success'].rolling(window=plot_window_size, min_periods=1).mean() * 100,
                           label=f'Mov Avg Success ({plot_window_size} eps)', color='orange', linestyle='--')
             ax0_twin.set_ylabel("Success Rate (%)")
             ax0_twin.legend(loc='lower right')

             # Loss Plot (from CSV - represents episode average)
             # Note: TensorBoard 'training/loss' will be per-step loss
             valid_loss_ep = log_data['AvgLossEp'].dropna()
             if not valid_loss_ep.empty:
                 axs[1].plot(log_data['Episode'][valid_loss_ep.index], valid_loss_ep, label='Avg Episode Loss (from CSV)', alpha=0.6, linewidth=0.8)
                 axs[1].plot(log_data['Episode'][valid_loss_ep.index], valid_loss_ep.rolling(window=plot_window_size // 5, min_periods=1).mean(),
                             label=f'Mov Avg Ep Loss ({plot_window_size // 5} eps)', color='green')
                 axs[1].set_ylabel("Loss (MSE)")
                 axs[1].set_title("Average Episode Training Loss (from CSV)")
                 axs[1].set_yscale('log') # Log scale often helpful for loss
                 axs[1].legend()
                 axs[1].grid(True)
             else:
                 axs[1].set_title("No valid episode loss data in CSV to plot.")

             # Epsilon Decay Plot
             axs[2].plot(log_data['Episode'], log_data['EpsilonEnd'], label='Epsilon (End of Episode)', color='purple')
             axs[2].set_ylabel("Epsilon Value")
             axs[2].set_title("Epsilon Decay Over Episodes")
             axs[2].legend()
             axs[2].grid(True)

             # Final IoW Plot
             axs[3].plot(log_data['Episode'], log_data['FinalIoW'], label='Final IoW per Episode', alpha=0.3, color='blue')
             axs[3].plot(log_data['Episode'], log_data['FinalIoW'].rolling(window=plot_window_size, min_periods=1).mean(),
                        label=f'Mov Avg Final IoW ({plot_window_size} eps)', color='cyan')
             axs[3].set_xlabel("Episode")
             axs[3].set_ylabel("IoW")
             axs[3].set_title("Final Intersection over Window (IoW) per Episode")
             axs[3].legend()
             axs[3].grid(True)


             plt.tight_layout()
             plt.savefig(PLOT_FILENAME)
             print(f"Training plot saved to {PLOT_FILENAME}")
             # plt.show()

        else:
            print(f"Log file {LOG_FILENAME} not found. Cannot generate plots.")

    except Exception as plot_e:
         print(f"Error generating plots: {plot_e}")
         traceback.print_exc()


if __name__ == "__main__":
    check_gpu()
    main()