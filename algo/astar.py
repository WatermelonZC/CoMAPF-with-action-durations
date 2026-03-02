import heapq
import numpy as np

def heuristic(a, b):
    """(row, col) 元组之间的曼哈顿距离。"""
    (r1, c1) = a
    (r2, c2) = b
    return abs(r1 - r2) + abs(c1 - c2)


# ----------------------------------------------------------------------
# Part 2: C++ 包装器
# ----------------------------------------------------------------------
def run_dijkstra(planner, start):
    """[C++ 包装器] 调用 C++ Dijkstra"""
    return planner.run_dijkstra(start)


def a_star_search(planner, start, goal):
    """[C++ 包装器] 调用 C++ A*"""
    path, cost = planner.a_star_search(start, goal)
    return path  # (返回路径)


def space_time_a_star(planner, start, goal, constraints,
                      start_time=0, max_time=800):
    """[C++ 包装器] 调用 C++ Space-Time A*"""
    # 确保所有键都存在，即使是空的
    if 'vertex' not in constraints: constraints['vertex'] = set()
    if 'edge' not in constraints: constraints['edge'] = set()
    if 'vertex_range' not in constraints: constraints['vertex_range'] = []

    return planner.space_time_a_star(start, goal, constraints,
                                     start_time, max_time)



# def space_time_a_star(grid_or_planner, start, goal, constraints,
#                       start_time=0, max_time=800):
#     # 检查我们是否在 C++ 模式
#     if FastPlanner is not None:
#         # [C++ 路径]
#         # (确保约束字典包含所有键，即使是空集/列表)
#         if 'vertex' not in constraints: constraints['vertex'] = set()
#         if 'edge' not in constraints: constraints['edge'] = set()
#         if 'vertex_range' not in constraints: constraints['vertex_range'] = []
#
#         return grid_or_planner.plan(start, goal, constraints,
#                                     start_time, max_time)
#     else:
#         # [回退 Python 路径]
#         # (确保您的 Python A* 实现了 'vertex_range' 检查!)
#         # print("警告: 正在使用慢速 Python A*")
#         return py_space_time_a_star(grid_or_planner, start, goal, constraints,
#                                     start_time, max_time)


# ----------------------------------------------------------------------
# Part 2: 时空 A* (Space-Time A* Search)
# 用于 CBTMP 低层策略的路径规划
# ----------------------------------------------------------------------
def py_space_time_a_star(grid, start, goal, constraints,
                      start_time=0, max_time=800):
    height, width = grid.shape

    initial_state = (start[0], start[1], start_time)  # (row, col, time)

    # (现在这些访问是安全的了)
    if grid[start[0], start[1]] == 1:
        # print(f"S-T A* 错误：起点 {start} 是障碍物") # (可选的调试信息)
        return []
    if grid[goal[0], goal[1]] == 1:
        # print(f"S-T A* 错误：终点 {goal} 是障碍物") # (可选的调试信息)
        return []

    open_set = []
    h_start = heuristic(start, goal)
    heapq.heappush(open_set, (h_start + start_time, start_time, initial_state))  # f, g, state
    came_from = {initial_state: None}

    vertex_cons = constraints.get('vertex', set())
    edge_cons = constraints.get('edge', set())
    neighbors_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # 5-向

    while open_set:
        current_f, current_g, current_state = heapq.heappop(open_set)
        current_pos = (current_state[0], current_state[1])
        current_time = current_state[2]

        if current_pos == goal:
            path_states = []
            temp = current_state
            while temp is not None:
                path_states.append(temp)
                temp = came_from.get(temp)
            path_states.reverse()
            spatial_path = [(r, c) for (r, c, t) in path_states]
            return spatial_path

        if current_time > max_time:
            continue

        for dr, dc in neighbors_deltas:
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)
            new_time = current_time + 1
            new_state = (new_pos[0], new_pos[1], new_time)

            if not (0 <= new_pos[0] < height and 0 <= new_pos[1] < width): continue
            if grid[new_pos[0], new_pos[1]] == 1: continue
            if new_state in came_from: continue
            if (new_pos, new_time) in vertex_cons: continue
            if ((current_pos, new_pos), new_time) in edge_cons: continue
            # if ((new_pos, current_pos), new_time) in edge_cons: continue

            came_from[new_state] = current_state
            g_score = new_time
            h_score = heuristic(new_pos, goal)
            f_score = g_score + h_score
            heapq.heappush(open_set, (f_score, g_score, new_state))
    return []

# ----------------------------------------------------------------------
# Part 1: 标准 A* (Standard A* Search)
# 用于 CBTMP 高层策略的成本预计算
# ----------------------------------------------------------------------
def a_star_search(grid, start, goal):
    """
    在 2D 网格上执行标准 A* 搜索。
    (来自我们之前的步骤)

    返回:
        (list, int): (path, cost)
    """
    height, width = grid.shape
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))  # f, g, pos
    came_from = {start: None}
    g_score = {pos: float('inf') for pos in np.ndindex(grid.shape)}
    g_score[start] = 0

    if not (0 <= start[0] < height and 0 <= start[1] < width): return [], float('inf')
    if not (0 <= goal[0] < height and 0 <= goal[1] < width): return [], float('inf')
    if grid[start[0], start[1]] == 1: return [], float('inf')
    if grid[goal[0], goal[1]] == 1: return [], float('inf')

    while open_set:
        _, current_g, current_pos = heapq.heappop(open_set)

        if current_pos == goal:
            path = []
            temp = current_pos
            while temp is not None:
                path.append(temp)
                temp = came_from[temp]
            path.reverse()
            return path, current_g

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_pos = (current_pos[0] + dr, current_pos[1] + dc)

            if not (0 <= neighbor_pos[0] < height and 0 <= neighbor_pos[1] < width):
                continue
            if grid[neighbor_pos[0], neighbor_pos[1]] == 1:
                continue

            new_g = current_g + 1
            if new_g < g_score.get(neighbor_pos, float('inf')):
                g_score[neighbor_pos] = new_g
                came_from[neighbor_pos] = current_pos
                f = new_g + heuristic(neighbor_pos, goal)
                heapq.heappush(open_set, (f, new_g, neighbor_pos))

    return [], float('inf')  # 未找到路径



if __name__ == '__main__':
    test_grid_str = """
    .........
    .@@@@@@@.
    .@@...@@.
    .@@.S.@@.
    .@@...@@.
    .@@.G.@@.
    .@@@@@@@.
    .........
    """
    test_grid = np.array([
        [1 if char == '@' else 0 for char in row]
        for row in test_grid_str.strip().split('\n')
    ])

    start_pos = (3, 4)  # 'S'
    goal_pos = (5, 4)   # 'G'

    # 2. 定义约束
    # 假设我们想强制 A* 绕路
    # 禁止它在 t=1 时访问 (4, 4) -- 这是最短路径上的一步
    constraints_example = {
        'vertex': { ((4, 4), 1) }, # 禁止 (row, col), time
        'edge': set()
    }

    print("测试时空 A* (有约束)...")
    path = space_time_a_star(test_grid, start_pos, goal_pos, constraints_example)

    if path:
        print(f"找到路径: {path}")
        print(f"路径长度: {len(path) - 1} 步")
    else:
        print("未找到路径。")

    print("\n测试时空 A* (无约束)...")
    path_no_constraints = space_time_a_star(test_grid, start_pos, goal_pos, {'vertex': set(), 'edge': set()})
    if path_no_constraints:
        print(f"找到路径: {path_no_constraints}")
        print(f"路径长度: {len(path_no_constraints) - 1} 步")