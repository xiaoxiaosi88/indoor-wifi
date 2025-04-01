# agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import random
from collections import deque
import os

import copy  # For deepcopy if needed inside remember (less crucial now)
# --- TB ---
from datetime import datetime  # For TensorBoard logging
# ----------

class DQNAgent:
    """ DQN Agent including normalization logic and TensorBoard support. """

    def __init__(self, state_input_dim, action_dim, history_len, norm_params,
                 learning_rate=0.0005, gamma=0.1,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=100, batch_size=50, target_update_freq=100,
                 log_dir="logs/default_run"): # --- TB: Added log_dir ---

        if state_input_dim <= 0: raise ValueError("state_input_dim must be positive.")
        self.state_input_dim = int(state_input_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.initial_epsilon = float(epsilon)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.learning_rate = float(learning_rate)
        self.target_update_freq = int(target_update_freq)
        self.learn_step_counter = 0

        # --- Normalization parameters handling ---
        self.norm_params = norm_params
        self.use_normalization = False
        if norm_params and isinstance(norm_params, dict):
            self.rssi_mean = float(norm_params.get('rssi_mean', 0.0))
            self.rssi_std = float(norm_params.get('rssi_std', 1.0))
            if abs(self.rssi_std) < 1e-6: self.rssi_std = 1.0
            self.coords_mean = np.array(norm_params.get('coords_mean', [0.0, 0.0]), dtype=np.float32)
            self.coords_std = np.array(norm_params.get('coords_std', [1.0, 1.0]), dtype=np.float32)
            self.coords_std[np.abs(self.coords_std) < 1e-6] = 1.0
            self.radius_mean = float(norm_params.get('radius_mean', 0.0))
            self.radius_std = float(norm_params.get('radius_std', 1.0))
            if abs(self.radius_std) < 1e-6: self.radius_std = 1.0
            self.use_normalization = True
            print("Agent: Normalization parameters loaded and enabled.")
            print(f"  RSSI: mean={self.rssi_mean:.2f}, std={self.rssi_std:.2f}")
            print(f"  Coords: mean={self.coords_mean}, std={self.coords_std}")
            print(f"  Radius: mean={self.radius_mean:.2f}, std={self.radius_std:.2f}")
        else:
            print("Warning: Normalization params not provided or invalid. Normalization disabled.")
        # -----------------------------------------

        # --- TB: Setup TensorBoard ---
        self.log_dir = log_dir
        print(f"Agent: Setting up TensorBoard logging in: {self.log_dir}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.train_step = 0 # Use this for logging training-step metrics
        # ---------------------------


        # Create models
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = losses.MeanSquaredError()
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()  # Initial weight copy


    def _build_model(self):
        """Builds the Neural Network Model."""
        model = models.Sequential([
            layers.InputLayer(input_shape=(self.state_input_dim,)),
            layers.Dense(2000, activation='relu'),
            layers.Dense(1448, activation='relu'),
            layers.Dense(548, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ], name="DQN_Model")
        # No need to compile if using GradientTape for training
        # model.compile(optimizer=self.optimizer, loss=self.loss_function)
        return model

    def _prepare_state_input(self, state):##将状态进行归一化
        """Converts state tuple to a standardized, flattened NumPy array."""
        if state is None: return np.zeros(self.state_input_dim, dtype=np.float32)

        rssi_vector, current_window, current_history = state
        if current_window is None or not isinstance(current_window, (list, tuple)) or len(
                current_window) != 2 or not isinstance(current_window[0], (list, tuple)) or len(current_window[0]) != 2:
            # print(f"Warning: Invalid window format in state: {current_window}. Returning zero vector.") # Reduce noise
            return np.zeros(self.state_input_dim, dtype=np.float32)

        center_coords, radius = current_window
        rssi_vector_np = np.array(rssi_vector, dtype=np.float32)
        window_vec = np.array([center_coords[0], center_coords[1], radius], dtype=np.float32)
        current_history_np = np.array(current_history, dtype=np.float32).reshape(-1)

        if self.use_normalization:
            rssi_vector_np = (rssi_vector_np - self.rssi_mean) / self.rssi_std
            window_vec[0:2] = (window_vec[0:2] - self.coords_mean) / self.coords_std
            window_vec[2] = (window_vec[2] - self.radius_mean) / self.radius_std

        try:
            flat_state = np.concatenate([rssi_vector_np, window_vec, current_history_np])
        except ValueError as e:
            print(f"Error concatenating state parts:")
            print(f"  RSSI shape: {rssi_vector_np.shape}, dtype: {rssi_vector_np.dtype}")
            print(f"  Window shape: {window_vec.shape}, dtype: {window_vec.dtype}")
            print(f"  History shape: {current_history_np.shape}, dtype: {current_history_np.dtype}")
            raise e

        if flat_state.shape[0] != self.state_input_dim:
             # This should ideally not happen if config is right, but good check
            print(f"Warning: Prepared state final dimension {flat_state.shape[0]} != expected {self.state_input_dim}. Padding/Truncating.")
            # Simple padding/truncating - might hide deeper issues
            correct_state = np.zeros(self.state_input_dim, dtype=np.float32)
            len_to_copy = min(flat_state.shape[0], self.state_input_dim)
            correct_state[:len_to_copy] = flat_state[:len_to_copy]
            flat_state = correct_state
            # Alternatively, raise ValueError:
            # raise ValueError(f"Prepared state final dimension {flat_state.shape[0]} != expected {self.state_input_dim}")


        if np.isnan(flat_state).any():
            print("Warning: NaN detected in prepared state vector. Replacing with zeros.")
            flat_state = np.nan_to_num(flat_state, nan=0.0, posinf=0.0, neginf=0.0) # Replace NaNs

        return flat_state ##输出归一化状态向量

    # --- TB: Logging Helper ---
    def log_metrics(self, metrics, step):
        """Log metrics to TensorBoard.

        Args:
            metrics (dict): Dictionary of metric names (str) and values (float/int).
            step (int): The step (e.g., training step or episode number) to associate.
        """
        with self.summary_writer.as_default():
            for name, value in metrics.items():
                if value is not None and not np.isnan(value):
                    try:
                         tf.summary.scalar(name, value, step=step)
                    except Exception as e:
                         print(f"Warning: Could not log metric '{name}' with value {value} at step {step}. Error: {e}")
            self.summary_writer.flush()
    # ------------------------

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the buffer."""
        if state is None or next_state is None:
            # print("Warning: Skipping remembrance due to None state(s).") # Reduce noise
            return
        # Deepcopy might be needed if env reuses state objects, but often not required
        self.memory.append((state, action, reward, next_state, done))##这里已经是滚动的了

    def act(self, state):
        """Selects action using epsilon-greedy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        prepared_state = self._prepare_state_input(state)
        # Handle rare case where preparation might fail and return zeros
        if np.all(prepared_state == 0):
             # print("Warning: Acting based on zero state vector.") # Reduce noise
             return random.randrange(self.action_dim) # Fallback to random action

        state_tensor = tf.convert_to_tensor([prepared_state], dtype=tf.float32)
        # Use tf.function for potential speedup, though predict is often fast enough
        # @tf.function
        # def predict_q(model, tensor):
        #     return model(tensor, training=False)
        # q_values = predict_q(self.model, state_tensor)[0].numpy()
        q_values = self.model.predict(state_tensor, verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):###梯度下降更新网络
        """Trains the main model using a batch from the replay buffer."""
        if len(self.memory) < self.batch_size: return None

        minibatch = random.sample(self.memory, self.batch_size)

        current_states_np = np.array([self._prepare_state_input(s) for s, a, r, ns, d in minibatch], dtype=np.float32)
        next_states_np = np.array([self._prepare_state_input(ns) for s, a, r, ns, d in minibatch], dtype=np.float32)
        actions = np.array([a for s, a, r, ns, d in minibatch], dtype=np.int32)
        rewards = np.array([r for s, a, r, ns, d in minibatch], dtype=np.float32)
        dones = np.array([d for s, a, r, ns, d in minibatch], dtype=np.float32)

        # Predict Q-values with target model
        future_q_values_target = self.target_model.predict(next_states_np, batch_size=self.batch_size, verbose=0)
        max_future_q = np.max(future_q_values_target, axis=1)
        target_q_values = rewards + self.gamma * max_future_q * (1.0 - dones)

        # Train main model using GradientTape
        with tf.GradientTape() as tape:
            q_values_main = self.model(current_states_np, training=True) # Get Q-values for all actions
            # Select the Q-value for the action actually taken
            action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), actions], axis=1)
            current_q_selected = tf.gather_nd(q_values_main, action_indices)
            # Calculate loss between selected Q-value and target Q-value
            loss = self.loss_function(target_q_values, current_q_selected)

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Clip gradients for stability (optional but often helpful)
        # gradients = [tf.clip_by_norm(g, 1.0) for g in gradients if g is not None] # Simple norm clipping
        valid_grads_vars = [(g, v) for g, v in zip(gradients, self.model.trainable_variables) if g is not None]

        if len(valid_grads_vars) < len(self.model.trainable_variables):
            print("Warning: Found None gradients during backprop.")
        if not valid_grads_vars:
            print("Error: No valid gradients found. Skipping optimizer step.")
            return loss.numpy() # Return loss even if step skipped

        self.optimizer.apply_gradients(valid_grads_vars)

        # --- TB: Log training metrics ---
        self.train_step += 1 # Increment training step counter
        metrics_to_log = {
            'training/loss': loss.numpy(),
            'training/epsilon': self.epsilon,
            'training/buffer_size': len(self.memory),
            # Calculate Q-value stats from the batch predictions
            'training/max_q_value': np.mean(np.max(q_values_main.numpy(), axis=1)),
            'training/mean_q_value': np.mean(q_values_main.numpy()),
            'training/target_q_mean': np.mean(target_q_values)
        }
        self.log_metrics(metrics_to_log, step=self.train_step)
        # ------------------------------

        # Update counters and target model potentially
        self.learn_step_counter += 1
        self._update_epsilon() # Epsilon decay tied to learning steps
        if self.learn_step_counter % self.target_update_freq == 0:
            self._update_target_model()
            print(f"Step {self.learn_step_counter}: Updated target network.") # Add confirmation

        return loss.numpy()

    def _update_target_model(self):
        """Copies weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())
        # --- TB: Log target model update event ---
        with self.summary_writer.as_default():
            tf.summary.text(
                'training/target_model_update',
                f'Updated target model at train_step {self.train_step}',
                step=self.train_step # Log against the training step count
            )
        # --------------------------------------

    def _update_epsilon(self):
        """Decays epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # Log epsilon here if you want it logged *every* training step
        # self.log_metrics({'training/epsilon_detailed': self.epsilon}, step=self.train_step)

    def load(self, name):
        """Loads model weights from a file."""
        if os.path.exists(name):
            try:
                self.model.load_weights(name)
                self._update_target_model()  # Sync target model
                print(f"Agent: Model weights loaded from {name}")
                # --- TB: Log model loading event ---
                with self.summary_writer.as_default():
                    tf.summary.text(
                        'model/load',
                        f'Loaded model weights from {name} at train_step {self.train_step}',
                         step=self.train_step # Log against the current train step
                    )
                # -----------------------------------
            except Exception as e:
                print(f"Agent Error: Could not load weights from {name}. Starting fresh. Error: {e}")
        else:
            print(f"Agent: Weight file {name} not found. Starting fresh.")

    def save(self, name):
        """Saves model weights to a file."""
        try:
            self.model.save_weights(name)
            # --- TB: Log model saving event ---
            # Don't print here, main loop does
            with self.summary_writer.as_default():
                 tf.summary.text(
                     'model/save',
                     f'Saved model weights to {name} at train_step {self.train_step}',
                     step=self.train_step # Log against the current train step
                 )
            # --------------------------------
        except Exception as e:
            print(f"Agent Error: Could not save weights to {name}. Error: {e}")