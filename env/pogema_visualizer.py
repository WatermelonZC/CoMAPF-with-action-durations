import os

from algo.CBTMP import CBTMP_Solver
from algo.CoCBS import CoCBS
from algo.pp import CooperativePriorityPlanning
from pogema import AnimationConfig, pogema_v0, AnimationMonitor, GridConfig
import time
from utils.path_validator import validate_scenario


def translate_paths_to_actions(paths_dict, num_agents, tasks, debug=False):
    """
    将 (CoCBS 或 CBTMP) 的坐标路径转换为 POGEMA 的动作序列。
    (此函数是通用的，无需更改)
    """
    if debug:
        print("\n--- [translate_paths_to_actions] 启动 ---")
        print(f"原始 paths_dict: {paths_dict}")

    def _get_action_from_move(pos_from, pos_to):
        dr, dc = pos_to[0] - pos_from[0], pos_to[1] - pos_from[1]
        if (dr, dc) == (0, 0): return 0  # No-op
        if (dr, dc) == (-1, 0): return 1  # Up
        if (dr, dc) == (1, 0): return 2  # Down
        if (dr, dc) == (0, -1): return 3  # Left
        if (dr, dc) == (0, 1): return 4  # Right
        return 0

    max_len = 0
    if not paths_dict: return []
    for path in paths_dict.values():
        max_len = max(max_len, len(path))
    if max_len <= 1: return []

    full_action_sequence = []
    agent_map = {}
    idx = 0
    num_tasks_in_env = num_agents // 2
    all_task_ids_from_solver = sorted(tasks.keys())
    task_ids_to_map = all_task_ids_from_solver[:num_tasks_in_env]

    for task_id in task_ids_to_map:
        agent_map[f'alpha_{task_id}'] = idx
        agent_map[f'beta_{task_id}'] = idx + 1
        idx += 2

    if debug: print(f"智能体映射 (agent_map): {agent_map}")

    for t in range(max_len - 1):
        actions_at_t_list = [0] * num_agents
        for agent_name, path in paths_dict.items():
            if agent_name not in agent_map: continue
            agent_index = agent_map[agent_name]
            pos_t = path[min(t, len(path) - 1)]
            pos_t_plus_1 = path[min(t + 1, len(path) - 1)]
            action = _get_action_from_move(pos_t, pos_t_plus_1)
            actions_at_t_list[agent_index] = action
        full_action_sequence.append(actions_at_t_list)

    if debug: print("\n--- [translate_paths_to_actions] 完成 ---")
    return full_action_sequence


def run_solver_in_pogema(solver_type, grid_np, tasks, max_episode_steps=999, save_filename="demo.svg", debug_translator=False, size=10,
                         conflict_heuristic=False, expand_root_heuristic=False, PP_heuristic_method="Longest Path First", PP_flexible=False,
                         use_heatmap=False, debug=False, Objective = 'Makespan'):
    """
    在 POGEMA 中运行并可视化 Co-CBS 或 CBTMP 求解器的结果。

    参数:
        solver_type (str): "CoCBS" 或 "CBTMP"
        grid_str (str): 描述地图的多行字符串。
        tasks (dict): 任务定义。
        save_filename (str): 保存动画的文件名 (例如 "my_run.svg")。
        debug_translator (bool): 是否为路径转换器启用详细日志。
    """

    print(f"--- 启动 {solver_type} 求解器 ---")

    # 1. 解析网格
    print("1. 正在解析网格...")
    # grid_np = parse_grid_from_str(grid_str)

    # --- [修改] 2. 根据选择实例化并运行求解器 ---
    print(f"2. {solver_type} 正在规划路径...")
    if solver_type.upper() == 'COCBS':
        # solver = Co_CBS_Solver(grid_np, tasks)
        pass
    elif solver_type.upper() == 'CBTMP':
        solver = CBTMP_Solver(grid_np, tasks,objective=Objective, max_sim_time=max_episode_steps, debug=debug)
    elif solver_type.upper() == 'SCBS':
        solver = CoCBS(grid_np, tasks, objective= Objective, max_sim_time=max_episode_steps, conflict_heuristic=conflict_heuristic, expand_root_heuristic=expand_root_heuristic, debug = debug)
    elif solver_type.upper() == 'PP':
        solver = CooperativePriorityPlanning(grid_np, tasks, objective=Objective, max_sim_time=max_episode_steps, heuristic_method=PP_heuristic_method,
                                             PP_flexible=PP_flexible, use_heatmap=use_heatmap, debug=debug)
    else:
        print(f"!! 错误：未知的求解器类型 '{solver_type}'。")
        return None, None

    start_time = time.time()
    paths = solver.solve()
    end_time = time.time()

    if not paths:
        print(f"!! {solver_type} 未能找到解。")
        return None, None

    # ==========================================
    # [新增] 快速合法性检查 (Validator Integration)
    # ==========================================
    print(f"--- 求解成功！(用时 {end_time - start_time:.4f} 秒) ---")
    print("\n>>> 正在执行路径合法性快速检查 (Path Validator)...")
    is_valid = validate_scenario(paths, tasks)

    if not is_valid:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!! [严重警告] 路径检查未通过！存在冲突或逻辑错误。")
        print("!! 请检查上方的 Validator 错误日志。动画演示可能不正确。")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        print(">>> [确认] 路径检查通过：无对穿、无撞车、逻辑合规。\n")
    # ==========================================

    print(f"--- 求解成功！(用时 {end_time - start_time:.4f} 秒) ---")
    for agent_id, path in sorted(paths.items()):
        print(f"  {agent_id}: {path}")

    # 3. 派生 POGEMA 环境配置
    # (此部分是通用的，无需修改)
    print("3. 正在配置 POGEMA 环境...")
    num_agents = len(tasks) * 2
    agent_starts_xy = []
    agent_targets_xy = []

    for task_id in sorted(tasks.keys()):
        task_data = tasks[task_id]
        agent_starts_xy.append(task_data['start_alpha'])
        agent_targets_xy.append(task_data['goal_alpha'])
        agent_starts_xy.append(task_data['start_beta'])
        agent_targets_xy.append(task_data['pickup_beta'])

    grid_config = GridConfig(
        map=grid_np.tolist(),
        num_agents=num_agents,
        agents_xy=agent_starts_xy,
        targets_xy=agent_targets_xy,
        max_episode_steps=9999,
        collision_system="ignore",
        on_target='nothing',
        size=size
    )

    # 4. 转换路径为动作
    # (此部分是通用的，无需修改)
    print("4. 正在转换路径为 POGEMA 动作...")
    action_sequence = translate_paths_to_actions(paths, num_agents, tasks, debug_translator)

    if not action_sequence:
        print("路径为空，无需执行。")
        return paths, None

    # 5. 在 POGEMA 中运行回放
    # (此部分是通用的，无需修改)
    print(f"5. 规划完成。开始在 POGEMA 中播放 {len(action_sequence)} 个动作...")
    env = pogema_v0(grid_config=grid_config)
    env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    obs, info = env.reset()
    for t, actions_at_t in enumerate(action_sequence):
        obs, rewards, terminated, truncated, info = env.step(actions_at_t)
        if all(terminated) or all(truncated):
            print(f"在时间步 {t} 结束。")
            break
    print("...播放完成。")

    # 6. 保存动画
    # (此部分是通用的，无需修改)
    print("6. 正在保存动画...")
    anim_folder = 'renders'
    if not os.path.exists(anim_folder):
        os.makedirs(anim_folder)

    animation_path = os.path.join(anim_folder, save_filename)
    env.save_animation(animation_path)
    env.close()

    print(f"--- 动画已保存到: {animation_path} ---")

    return paths, animation_path
