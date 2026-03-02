import numpy as np
from collections import deque


def generate_congestion_heatmap(grid, tasks):
    """
    生成静态拥堵热力图。
    原理: 模拟所有任务的最短路径流，统计每个格子的被访问频率。
    """
    rows, cols = grid.shape
    heatmap = np.zeros((rows, cols), dtype=np.float32)

    # 简单的 BFS 寻路来估算流量
    def get_shortest_path_cells(start, goal):
        q = deque([(start, [start])])
        visited = {start}
        while q:
            curr, path = q.popleft()
            if curr == goal:
                return path

            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append(((nr, nc), path + [(nr, nc)]))
        return []

    # print(">>> 正在生成交通热力图 (基于任务流)...")

    count = 0
    for task_id, task in tasks.items():
        # Alpha 任务
        s_a, g_a = task['start_alpha'], task['goal_alpha']
        path_a = get_shortest_path_cells(s_a, g_a)
        for r, c in path_a:
            heatmap[r, c] += 0.1

        # Beta 任务
        s_b, p_b = task['start_beta'], task['pickup_beta']
        path_b = get_shortest_path_cells(s_b, p_b)
        for r, c in path_b:
            heatmap[r, c] += 0.1

        count += 1
        # 采样一部分任务即可，不必全部跑完，太慢
        if count > 50:
            break

    # 归一化: 将热力值映射到 0.0 ~ 5.0 之间作为惩罚 Cost
    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap)) * 0.1  # 最高惩罚 10

    # 也可以对墙壁周边进行膨胀 (Dilation)，让智能体倾向于走大路
    return heatmap.tolist()  # 转为 list 传给 C++