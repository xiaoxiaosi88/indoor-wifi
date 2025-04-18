import numpy as np
import copy
from environment import SingleFloorLocalizationEnv

class MultiTargetLocalizationEnv(SingleFloorLocalizationEnv):
    """
    在 SingleFloorLocalizationEnv 基础上，支持一次 reset 多个 ground_truth，
    并为 Option-Critic 提供"子目标完成(done_low)"和"所有子目标完成(done_high)"信号。
    """
    def reset(self, rssi_vector, ground_truth_locs: list, bounds):
        """
        ground_truth_locs: [(x1,y1), (x2,y2), ...] 多个目标
        """
        # 调用父类部分逻辑，仅初始化 RSSI、bounds、step 计数
        if not isinstance(ground_truth_locs, list) or not ground_truth_locs:
            print("Error: ground_truth_locs must be a non-empty list of tuples.")
            return None

        # 复用父类对单目标的检查
        state = super().reset(rssi_vector, ground_truth_locs[0], bounds)
        if state is None:
            return None

        # 额外的多目标管理
        self.ground_truth_locs = ground_truth_locs
        self.current_target_idx = 0
        self.num_targets = len(ground_truth_locs)
        # 保存 low‐level 完成标志和 high‐level 完成标志
        return (self.rssi_vector, self.current_window, self.current_history,
                {"done_low": False, "done_high": False})

    def step(self, action):
        """
        执行一个低层动作，返回 low‐level state, reward_low, done_low, info_low,
                       high‐level done_high (所有目标完成)。
        """
        # 先复用父类逻辑
        next_state, reward, done, info = super().step(action)
        done_low = done   # 子目标完成或失败

        done_high = False
        # 如果子目标刚刚成功 (reward == +eta) 或失败，切换到下一个目标
        if done_low:
            if reward > 0:
                info['subgoal_success'] = True
            else:
                info['subgoal_success'] = False
            # 如果还没遍历完全部目标，切换到下一个子目标
            self.current_target_idx += 1
            if self.current_target_idx < self.num_targets:
                # 重置窗口，保留 rssi_vector、bounds 不变
                gt = self.ground_truth_locs[self.current_target_idx]
                self.ground_truth_window = [(gt[0], gt[1]), self.radgt]
                self.current_window = copy.deepcopy(self.initial_window)
                self.current_history = np.zeros(self.history_vec_size, dtype=np.float32)
                self.current_step = 0
                done_high = False
                # low‐level 的 done=False，继续给 low‐level agent 训练
                done_low = False
                reward = 0.0  # 或者给一个“切换目标”奖励/惩罚
            else:
                # 所有子目标都完成，episode 结束
                done_high = True

        # 将 done_high 汇报给上层 Option‐Critic
        info['done_high'] = done_high
        info['done_low'] = done_low
        # 返回结构也可以加层次信息
        return ( (self.rssi_vector, self.current_window, self.current_history),
                 float(reward),
                 done_low,
                 info )