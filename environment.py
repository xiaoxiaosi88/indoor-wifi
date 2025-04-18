# environment.py
import numpy as np
import copy

class SingleFloorLocalizationEnv:
    """
    Simulates the single-floor indoor localization environment based on
    the bisection RL approach.
    """
    def __init__(self, radgt=0.5, delta=0.5, eta=3, tau=1, max_steps=15, history_len=10, action_dim=4):
        if not (0 < delta <= 1): raise ValueError("delta must be in (0, 1]")
        self.radgt = float(radgt)
        self.delta = float(delta)
        self.eta = float(eta)
        self.tau = float(tau)
        self.max_steps = int(max_steps)
        self.history_len = int(history_len)
        self.action_dim = int(action_dim)
        self.history_vec_size = self.action_dim * self.history_len

        self.rssi_vector = None
        self.ground_truth_loc = None
        self.ground_truth_window = None
        self.bounds = None
        self.initial_window = None
        self.current_window = None
        self.current_history = None
        self.current_step = 0
        self._action_mapping = { 0: 'NW', 1: 'NE', 2: 'SW', 3: 'SE' } # Assuming 4 actions

    def _calculate_initial_window(self):
        if self.bounds is None: raise ValueError("Environment bounds not set.")
        xmin, xmax = self.bounds['xmin'], self.bounds['xmax']
        ymin, ymax = self.bounds['ymin'], self.bounds['ymax']
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        max_span = max(xmax - xmin, ymax - ymin)
        if max_span <= 0: raise ValueError(f"Invalid map bounds span: {max_span}. Bounds: {self.bounds}")
        # Add radgt to ensure the target isn't exactly on the boundary initially
        radius = (max_span / 2.0) + self.radgt
        return [(center_x, center_y), radius]


    def _calculate_iow(self, window1, window2):
        if window1 is None or window2 is None: return 0.0
        (cx1, cy1), rad1 = window1
        (cx2, cy2), rad2 = window2
        # Ensure radii are positive for valid area calculations
        if rad1 <= 1e-9 or rad2 <= 1e-9: return 0.0 # Treat very small radius as zero area

        w1_xmin, w1_xmax = cx1 - rad1, cx1 + rad1
        w1_ymin, w1_ymax = cy1 - rad1, cy1 + rad1
        w2_xmin, w2_xmax = cx2 - rad2, cx2 + rad2
        w2_ymin, w2_ymax = cy2 - rad2, cy2 + rad2

        inter_xmin = max(w1_xmin, w2_xmin)
        inter_ymin = max(w1_ymin, w2_ymin)
        inter_xmax = min(w1_xmax, w2_xmax)
        inter_ymax = min(w1_ymax, w2_ymax)

        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        intersection_area = inter_width * inter_height
        window1_area = (2 * rad1) ** 2 # Area of the predicted window

        # Prevent division by zero or near-zero
        if window1_area < 1e-9: return 0.0

        iow = intersection_area / window1_area
        return np.clip(iow, 0.0, 1.0) # Clamp between 0 and 1

    def _update_history(self, history_vec, action):##这里就是独热编码
        new_history = np.roll(history_vec, self.action_dim)
        one_hot_action = np.zeros(self.action_dim, dtype=np.float32)
        if 0 <= action < self.action_dim:
            one_hot_action[action] = 1.0
        else:
             print(f"Warning: Invalid action {action} provided to _update_history.")
        new_history[:self.action_dim] = one_hot_action
        return new_history

    def _calculate_next_window(self, current_window, action):
        (cx, cy), rad = current_window
        # Reduce radius, ensuring it doesn't become too small (or zero)
        next_rad = max(rad / 2.0, 1e-6)
        offset = next_rad # Distance to shift center based on new half-size

        if action == 0: next_cx, next_cy = cx - offset, cy + offset # 左上
        elif action == 1: next_cx, next_cy = cx + offset, cy + offset # 右上
        elif action == 2: next_cx, next_cy = cx - offset, cy - offset # 左下
        elif action == 3: next_cx, next_cy = cx + offset, cy - offset # 右下
        else:
            # Fallback for safety, although action should be validated before calling this
            print(f"Warning: Invalid action {action} in _calculate_next_window. Defaulting to NW.")
            next_cx, next_cy = cx - offset, cy + offset
        return [(next_cx, next_cy), next_rad]

    def reset(self, rssi_vector, ground_truth_loc, bounds):
        """ Resets the environment for a new episode. Returns initial state or None on failure."""
        # Input validation
        if not isinstance(bounds, dict) or not all(k in bounds for k in ['xmin', 'xmax', 'ymin', 'ymax']):
            print(f"Error: Invalid bounds format: {bounds}")
            return None
        if not isinstance(ground_truth_loc, tuple) or len(ground_truth_loc) != 2:
            print(f"Error: Invalid ground_truth_loc format: {ground_truth_loc}")
            return None
        if rssi_vector is None:
            print("Error: rssi_vector cannot be None for reset.")
            return None

        self.rssi_vector = np.array(rssi_vector, dtype=np.float32)##每一行RSSI的信号强度
        self.ground_truth_loc = ground_truth_loc
        self.bounds = bounds
        self.current_step = 0
        gt_x, gt_y = self.ground_truth_loc
        self.ground_truth_window = [(gt_x, gt_y), self.radgt]##以第一个点的经度和纬度坐标半径为0.5

        try:
            self.initial_window = self._calculate_initial_window()
            self.current_window = copy.deepcopy(self.initial_window)
        except ValueError as e:
            print(f"Error during env reset (initial window calc): {e}")
            return None # Indicate failure

        self.current_history = np.zeros(self.history_vec_size, dtype=np.float32)
        return (self.rssi_vector, self.current_window, self.current_history)

    def step(self, action):
        """ Executes one step. Returns (next_state, reward, done, info). """
        ##验证智能体动作是否有效，并在无效时终止回合
        if not (0 <= action < self.action_dim):
            print(f"Error: Invalid action {action} received in step.")
            # Return current state and terminate with penalty
            reward = -float(self.eta)
            done = True
            current_state = (self.rssi_vector, self.current_window, self.current_history) if self.current_window else None
            info = {'iow': 0.0, 'step': self.current_step + 1, 'error': 'Invalid action'}
            return current_state, reward, done, info

        if self.rssi_vector is None or self.current_window is None or self.ground_truth_window is None:
            raise RuntimeError("Environment not properly reset before calling step().")

        iow_current = self._calculate_iow(self.current_window, self.ground_truth_window)
        next_window = self._calculate_next_window(self.current_window, action)
        next_history = self._update_history(self.current_history, action)
        iow_next = self._calculate_iow(next_window, self.ground_truth_window)

        done = False
        reward = 0.0

        # Strict Reward Logic:
        if iow_next >= self.delta:
            reward = float(self.eta)  # Success reward
            done = True
        elif iow_next > iow_current:
            reward = float(self.tau)   # Improvement reward
            done = False
        else: # IoW <= current (stalled, decreased, or target lost)
            reward = -float(self.eta) # Failure penalty
            done = True

        self.current_step += 1
        # Max steps check: only overrides if not already terminated by IoW rules
        if not done and self.current_step >= self.max_steps:
            done = True
            # If timeout occurs, no explicit reward change unless desired
            # Current logic means if the last step had reward=tau, it keeps it.
            # If stricter penalty for timeout needed, set reward = -self.eta here too.
            # Setting reward=0.0 seems reasonable for timeout without success/failure.
            if reward == 0.0: reward = 0.0

        # Update internal state for next step
        self.current_window = next_window
        self.current_history = next_history
        next_state = (self.rssi_vector, self.current_window, self.current_history)
        info = {'iow': iow_next, 'step': self.current_step} # Include final IoW and step count
        return next_state, float(reward), done, info