# ----------------------------------------------------------------------
# [新] Part 1: 地图加载器
# ----------------------------------------------------------------------
import os
import random
from collections import deque

import numpy as np

class MapLoader:
    @staticmethod
    def load_map_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "map":
                start_idx = i + 1
                break
        map_lines = [l.strip() for l in lines[start_idx:] if l.strip()]
        height = len(map_lines)
        width = len(map_lines[0])
        grid_np = np.zeros((height, width), dtype=int)
        grid_str = ""
        for r, row_str in enumerate(map_lines):
            processed_row = ""
            for c, char in enumerate(row_str):
                if char in ['.', 'G', 'S', 'T']:
                    grid_np[r, c] = 0
                    processed_row += "."
                else:
                    grid_np[r, c] = 1
                    processed_row += "@"
            grid_str += processed_row + "\n"
        return grid_np, grid_str

    @staticmethod
    def get_largest_component(grid_np):
        height, width = grid_np.shape
        visited = np.zeros((height, width), dtype=bool)
        largest_comp = []
        for r in range(height):
            for c in range(width):
                if grid_np[r, c] == 0 and not visited[r, c]:
                    current_comp = []
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    current_comp.append((r, c))
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                if grid_np[nr, nc] == 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                                    current_comp.append((nr, nc))
                    if len(current_comp) > len(largest_comp):
                        largest_comp = current_comp
        return largest_comp

    @staticmethod
    def generate_reachable_tasks(valid_coords, num_tasks, duration_range=(1, 4)):
        """
        生成任务。
        :param duration_range: tuple (min, max), 默认 (1, 4) 保持原有逻辑不变。
        """
        if len(valid_coords) < num_tasks * 4:
            raise ValueError(f"连通区域太小 ({len(valid_coords)}), 无法生成 {num_tasks} 个任务")

        samples = random.sample(valid_coords, num_tasks * 4)
        tasks = {}
        idx = 0

        min_d, max_d = duration_range  # 解包时间范围

        for i in range(num_tasks):
            tasks[i] = {
                'start_alpha': samples[idx],
                'goal_alpha': samples[idx + 1],
                'start_beta': samples[idx + 2],
                'pickup_beta': samples[idx + 3],
                # 使用传入的范围
                'co_work_duration': random.randint(min_d, max_d),
                'pickup_duration': random.randint(min_d, max_d),
                'delivery_duration': random.randint(min_d, max_d)
            }
            idx += 4
        return tasks