# --- mapf_utils.py ---


def calculate_soc(paths):
    """
    计算总成本 (Sum-of-Costs)。
    (即您原来的 calculate_cost)
    """
    total_cost = 0
    for path in paths.values():
        total_cost += len(path) - 1
    return total_cost


def calculate_makespan(paths):
    """
        [新] 计算全局 Makespan (最晚完工时间)。
        这是您新要求的目标函数。
        """
    if not paths:
        return 0
    max_len = 0
    for path in paths.values():
        # 路径长度 - 1 = 任务完成时刻
        max_len = max(max_len, len(path) - 1)
    return max_len


# def _is_safe_to_stay(location, arrival_time, constraints, max_check_time=float('inf')):
#     """
#     检查在 arrival_time 到达 location 后，
#     停留到 max_check_time 是否违反 'vertex' 约束。
#
#     Args:
#         location (tuple): 要检查的停留位置
#         arrival_time (int): 到达该位置的时间
#         constraints (dict): 该智能体的约束字典 {'vertex': set(...), ...}
#         max_check_time (float): 检查的时间上限 (闭区间)
#
#     Returns:
#         bool: 如果停留安全则为 True，否则为 False
#     """
#     cons_set = constraints.get('vertex', set())
#     for (loc, t) in cons_set:
#         # 我们只关心在 arrival_time *之后* 到 max_check_time *之前* 的约束
#         if loc == location and t > arrival_time and t <= max_check_time:
#             # 冲突！在 t 时刻有一个约束，
#             # 而 agent 会在 t 时刻停在这里。
#             return False
#     # 没有发现冲突
#     return True


# def find_conflict(paths, intentional_meetings=None, pair_exemptions=None):
#     """
#     通用的 Co-MAPF 冲突查找器。
#     [来自 CoCBS.py]
#
#     它可以选择性地忽略在 `intentional_meetings` 中定义的*有意*冲突。
#
#     Args:
#         paths (dict): {agent_id: [path]}
#         intentional_meetings (dict, optional): {(pos, time) -> set(agents)}
#                                               定义了在特定时空点允许的汇合。
#
#     Returns:
#         dict or None: 描述冲突的字典，或 None (无冲突)
#     """
#     if intentional_meetings is None:
#         intentional_meetings = {}
#
#     max_len = max(len(p) for p in paths.values()) if paths else 0
#     if max_len == 0:
#         return None
#
#     # 1. 检查顶点冲突
#     locations_at_time = {}  # (pos, time) -> [agent_id]
#
#     for agent_id, path in paths.items():
#         for t in range(max_len):
#             # 如果路径较短，智能体停留在其终点
#             pos = path[min(t, len(path) - 1)]
#
#             key = (pos, t)
#             if key not in locations_at_time:
#                 locations_at_time[key] = []
#             locations_at_time[key].append(agent_id)
#
#     for (pos, time), agents in locations_at_time.items():
#         if len(agents) > 1:
#
#             # --- 豁免检查 1 (时间相关的：同步工作) ---
#             if (pos, time) in intentional_meetings:
#                 allowed_agents = intentional_meetings[(pos, time)]
#                 if all(a in allowed_agents for a in agents):
#                     continue
#
#             # --- [创新点 4 (新)] 豁免检查 2 (地点相关的：v_m 上的协作对) ---
#             if pos in pair_exemptions:
#                 allowed_pair = pair_exemptions[pos]
#
#                 # 检查是否 *所有* 发生冲突的智能体
#                 # 都属于这个被豁免的协作对
#                 all_agents_in_pair = True
#                 for a in agents:
#                     if a not in allowed_pair:
#                         all_agents_in_pair = False
#                         break
#
#                 if all_agents_in_pair:
#                     # 冲突只发生在 alpha_i 和 beta_i 之间，
#                     # 并且发生在他们的 v_m 点。
#                     continue  # 豁免这个冲突
#
#             # 如果不是被允许的汇合，则为冲突
#             return {'type': 'vertex', 'time': time, 'loc': pos, 'agents': agents}
#
#     # 2. 检查边 (交换) 冲突
#     for agent1_id, path1 in paths.items():
#         for agent2_id, path2 in paths.items():
#             if agent1_id >= agent2_id:
#                 continue
#
#             for t in range(max_len - 1):
#                 pos1_t0 = path1[min(t, len(path1) - 1)]
#                 pos1_t1 = path1[min(t + 1, len(path1) - 1)]
#
#                 pos2_t0 = path2[min(t, len(path2) - 1)]
#                 pos2_t1 = path2[min(t + 1, len(path2) - 1)]
#
#                 if (pos1_t1 == pos2_t0 and
#                         pos1_t0 == pos2_t1 and
#                         pos1_t0 != pos2_t0):
#                     return {'type': 'edge', 'time': t + 1,
#                             'loc1': pos1_t0, 'loc2': pos1_t1,
#                             'agents': [agent1_id, agent2_id]}
#
#     return None  # 没有冲突

## CPP调用
def _find_conflict(planner, paths, intentional_meetings, pair_exemptions):
    """[C++ 包装器] 调用 C++ Conflict Finder"""
    # 将 Python paths (list of tuples) 转换为 C++ (map of vectors)
    # (注意：pybind11 会自动处理这种转换)
    conflict_dict = planner.find_conflict(paths, intentional_meetings, pair_exemptions)

    # 将 C++ 返回的空 dict 转换为 Python None
    if not conflict_dict:
        return None
    return conflict_dict



## CPP调用
def _get_conflict_count(planner, paths, intentional_meetings):
    """
    [已修改] 计算给定路径和汇合点集合中的所有冲突。
    [已优化] 现在直接调用 C++ 核心函数。
    """
    return planner.count_all_conflicts(
        paths,
        intentional_meetings,
        dict()
    )

## CPP调用
def _is_safe_to_stay(location, arrival_time, constraints, max_check_time=float('inf')):
    """
    检查在 arrival_time 到达 location 后，
    停留到 max_check_time 是否违反 'vertex' 约束。

    Args:
        location (tuple): 要检查的停留位置
        arrival_time (int): 到达该位置的时间
        constraints (dict): 该智能体的约束字典 {'vertex': set(...), ...}
        max_check_time (float): 检查的时间上限 (闭区间)

    Returns:
        bool: 如果停留安全则为 True，否则为 False
    """
    cons_set = constraints.get('vertex', set())
    for (loc, t) in cons_set:
        # 我们只关心在 arrival_time *之后* 到 max_check_time *之前* 的约束
        if loc == location and t > arrival_time and t <= max_check_time:
            # 冲突！在 t 时刻有一个约束，
            # 而 agent 会在 t 时刻停在这里。
            return False
    # 没有发现冲突
    return True

def new_is_safe_to_stay(planner, location, arrival_time, constraints, max_check_time=-1):
    """
    [C++ 包装器] 调用 C++ is_safe_to_stay() 函数。
    """
    return planner.is_safe_to_stay(location, arrival_time, constraints, max_check_time)
