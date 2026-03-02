import copy
import heapq
import time
import numpy as np
from pogema_toolbox.registry import ToolboxRegistry

# [修改] 1. 导入 C++ 库
try:
    from algo.ecocbs_cpp_lib import EcoCBS_Planner
except ImportError:
    print("=" * 50)
    print("错误: 无法导入 C++ 核心 'ecocbs_cpp_lib'。")
    print("请确保您已经使用 setup.py 成功编译了 ECoCBS_cpp.cpp")
    print("=" * 50)
    exit(1)

from env.ctnode import Co_CBS_CTNode
from utils.file_util import custom_deep_copy_constraints
from utils.mapf_utils import _is_safe_to_stay, calculate_makespan, calculate_soc, _get_conflict_count, _find_conflict, new_is_safe_to_stay


class MeetingsTable:
    """
    实现 Co-CBS 的 Algorithm 2...
    [已修正] 确保成本计算遵循 Co-MAPF 路径定义。
    """

    def __init__(self, planner, task_id, task_data, objective):
        """
        [修改] 接收 C++ planner, 而不是 grid
        """
        self.planner = planner
        self.task_id = task_id
        self.task = task_data
        self.heap = []
        self.dists_goal_to_all = {}
        self.objective = objective
        self.pickup_duration = self.task.get('pickup_duration', 0)

        # --- [修正 Beta] ---
        # 1. [修改] 使用 C++ a_star_search
        path_b_pre, self.cost_b_pre = self.planner.a_star_search(self.task['start_beta'], self.task['pickup_beta'])
        if not path_b_pre:
            raise ValueError(f"任务 {self.task_id} (Beta) 无法从 {self.task['start_beta']} 到达 {self.task['pickup_beta']}")

        # --- [修正 Alpha] ---
        # 2. [修改] 使用 C++ run_dijkstra
        self.dists_alpha_start_to_all = self.planner.run_dijkstra(self.task['start_alpha'])

        # 3. [修改] 使用 C++ run_dijkstra
        self.dists_s_to_all = self.planner.run_dijkstra(self.task['pickup_beta'])
        self.dists_goal_to_all = self.planner.run_dijkstra(self.task['goal_alpha'])

        # 4. 填充汇合表 (逻辑不变)
        all_vertices = set(self.dists_alpha_start_to_all.keys()) | set(self.dists_s_to_all.keys())
        for v in all_vertices:
            if (v not in self.dists_alpha_start_to_all or
                    v not in self.dists_s_to_all or
                    v not in self.dists_goal_to_all):
                continue

            t_alpha = self.dists_alpha_start_to_all[v]
            cost_s_to_v = self.dists_s_to_all[v]

            if t_alpha == float('inf') or cost_s_to_v == float('inf'): continue

            # Beta 到达时间
            t_beta = self.cost_b_pre + self.pickup_duration + cost_s_to_v

            # 汇合开始时间
            t_star_v = max(t_alpha, t_beta)

            cost_v_g = self.dists_goal_to_all[v]
            if cost_v_g == float('inf'): continue

            base_cost = 2 * t_star_v + cost_v_g

            # 初始推入堆，此时 real_cost = base_cost
            heapq.heappush(self.heap, (base_cost, v, t_star_v))

    def get_next_meeting(self):
        if not self.heap:
            return None, float('inf')
        (cost, v_m, t_m) = heapq.heappop(self.heap)
        t_m_next = t_m + 1
        cost_v_g = self.dists_goal_to_all.get(v_m, float('inf'))
        if cost_v_g == float('inf'):
            return (v_m, t_m), cost
        cost_next = t_m_next + cost_v_g
        heapq.heappush(self.heap, (cost_next, v_m, t_m_next))
        return (v_m, t_m), cost

class CoCBS:
    def __init__(self, grid, tasks, max_sim_time=800, objective="Makespan", conflict_heuristic=True, expand_root_heuristic=True, debug=False, time_limit=None):
        self.grid = grid
        self.tasks = tasks
        self.open_roots = []
        self.open_regular = []
        self.meeting_tables = {}
        self.iteration = 0
        self.objective = objective
        self.time_limit = time_limit  # 保存时间限制
        self.expand_root_heuristic = expand_root_heuristic
        self.planner = EcoCBS_Planner(self.grid, max_sim_time, debug)
        self.conflict_heuristics = conflict_heuristic
        if self.objective == "Makespan":
            self.calculate_solution_cost = calculate_makespan
        else:
            self.calculate_solution_cost = calculate_soc

    def solve(self):
        start_time = time.time()
        try:
            for task_id, task_data in self.tasks.items():
                if self.time_limit and (time.time() - start_time) > self.time_limit:
                    return None

                self.meeting_tables[task_id] = MeetingsTable(self.planner, task_id, task_data, self.objective)
        except ValueError as e:
            print(f"!! {e}")
            return None
        root_constraints = {}
        root_meetings = {}
        for task_id, table in self.meeting_tables.items():
            meeting, cost = table.get_next_meeting()
            if meeting is None:
                return None
            root_meetings[task_id] = meeting
            root_constraints[f"alpha_{task_id}"] = {'vertex': set(), 'edge': set()}
            root_constraints[f"beta_{task_id}"] = {'vertex': set(), 'edge': set()}
        root_paths, success = self._plan_all_paths(root_meetings, root_constraints)
        if not success:
            ToolboxRegistry.info(f"!! 求解失败：无法为根节点 $\mathcal{root_meetings}^*$ 规划初始路径。")
            return None
        root_cost = self.calculate_solution_cost(root_paths)
        intentional = self._get_intentional_meetings_with_wildcard(root_meetings)
        root_conflict_count = 0
        if self.conflict_heuristics:
            root_conflict_count = _get_conflict_count(self.planner, root_paths, intentional)
        root_node = Co_CBS_CTNode(root_constraints, root_paths, root_cost, root_meetings, is_root=True, conflict_count=root_conflict_count)
        heapq.heappush(self.open_roots, root_node)
        while self.open_roots or self.open_regular:
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                return None
            self.iteration += 1
            if self.open_regular and (not self.open_roots or self.open_regular[0] < self.open_roots[0]):
                P = heapq.heappop(self.open_regular)
            else:
                P = heapq.heappop(self.open_roots)
            intentional = self._get_intentional_meetings_with_wildcard(P.meetings)
            conflict = _find_conflict(self.planner, P.paths, intentional, {})
            # 7. 检查解...
            if conflict is None:
                return P.paths
            if P.is_root:
                if self.expand_root_heuristic:
                    self._heuristic_expand_root(P, conflict)
                else:
                    self._expand_root(P)

            # 9. 分裂...
            agents_in_conflict = conflict['agents']
            for agent_to_constrain in agents_in_conflict:
                new_constraints = custom_deep_copy_constraints(P.agent_constraints)
                if conflict['type'] == 'vertex':
                    con = (conflict['loc'], conflict['time'])
                    new_constraints[agent_to_constrain]['vertex'].add(con)
                elif conflict['type'] == 'edge':
                    if agent_to_constrain == agents_in_conflict[0]:
                        con = ((conflict['loc1'], conflict['loc2']), conflict['time'])
                    else:
                        con = ((conflict['loc2'], conflict['loc1']), conflict['time'])
                    new_constraints[agent_to_constrain]['edge'].add(con)
                new_paths = P.paths.copy()
                new_meetings = P.meetings
                agent_type, task_id = agent_to_constrain.split('_')
                task_id = int(task_id)
                new_paths_group, success = self._plan_group(
                    task_id, new_meetings[task_id], new_constraints
                )
                if success:
                    new_paths.update(new_paths_group)
                    conflict_count = 0
                    new_intentional = self._get_intentional_meetings_with_wildcard(new_meetings)
                    if self.conflict_heuristics:
                        conflict_count = _get_conflict_count(self.planner, new_paths, new_intentional)
                    new_cost = self.calculate_solution_cost(new_paths)
                    new_node = Co_CBS_CTNode(new_constraints, new_paths, new_cost,
                                             new_meetings, is_root=False, conflict_count=conflict_count)
                    heapq.heappush(self.open_regular, new_node)
                else:
                    pass
        return None


    def _expand_root(self, P):
        for task_id_to_change in self.tasks.keys():
            R_constraints = {}
            R_meetings = P.meetings.copy()
            for t_id in self.tasks.keys():
                # [修改] 移除 'vertex_range'
                R_constraints[f"alpha_{t_id}"] = {'vertex': set(), 'edge': set()}
                R_constraints[f"beta_{t_id}"] = {'vertex': set(), 'edge': set()}

            next_meeting, next_cost_component = self.meeting_tables[task_id_to_change].get_next_meeting()
            if next_meeting is None: continue
            R_meetings[task_id_to_change] = next_meeting
            R_paths, success = self._plan_all_paths(R_meetings, R_constraints)
            if success:
                R_cost = self.calculate_solution_cost(R_paths)
                intentional_meetings = {}
                R_conflict_count = 0
                if self.conflict_heuristics:
                    for task_id, (v_m, t_m) in R_meetings.items():
                        pair = {f"alpha_{task_id}", f"beta_{task_id}"}
                        task_duration = self.tasks[task_id].get('co_work_duration', 0)
                        final_t_m = t_m + task_duration
                        for t in range(t_m, final_t_m + 1):
                            intentional_meetings[(v_m, t)] = pair
                    R_conflict_count = _get_conflict_count(self.planner, R_paths, intentional_meetings)
                R_node = Co_CBS_CTNode(R_constraints, R_paths, R_cost,
                                       R_meetings, is_root=True, conflict_count=R_conflict_count)
                heapq.heappush(self.open_roots, R_node)


    def _heuristic_expand_root(self, P, conflict):
        tasks_to_expand = set()
        agents_in_conflict = conflict.get('agents', [])

        for agent_id in agents_in_conflict:
            agent_type, task_id = agent_id.split('_')
            if task_id in self.tasks:
                tasks_to_expand.add(task_id)
        if not tasks_to_expand:
            tasks_to_expand = self.tasks.keys()
        else:
            pass

        for task_id_to_change in tasks_to_expand:
            R_constraints = {}
            R_meetings = P.meetings.copy()
            for t_id in self.tasks.keys():
                R_constraints[f"alpha_{t_id}"] = {'vertex': set(), 'edge': set()}
                R_constraints[f"beta_{t_id}"] = {'vertex': set(), 'edge': set()}
            next_meeting, next_cost_component = self.meeting_tables[task_id_to_change].get_next_meeting()
            if next_meeting is None: continue
            R_meetings[task_id_to_change] = next_meeting
            R_paths, success = self._plan_all_paths(R_meetings, R_constraints)
            if success:
                R_cost = self.calculate_solution_cost(R_paths)
                intentional_meetings = {}
                R_conflict_count = 0
                if self.conflict_heuristics:
                    for task_id, (v_m, t_m) in R_meetings.items():
                        pair = {f"alpha_{task_id}", f"beta_{task_id}"}
                        task_duration = self.tasks[task_id].get('co_work_duration', 0)
                        final_t_m = t_m + task_duration
                        for t in range(t_m, final_t_m + 1):
                            intentional_meetings[(v_m, t)] = pair
                    R_conflict_count = _get_conflict_count(self.planner, R_paths, intentional_meetings)
                R_node = Co_CBS_CTNode(R_constraints, R_paths, R_cost,
                                       R_meetings, is_root=True,
                                       conflict_count=R_conflict_count)
                heapq.heappush(self.open_roots, R_node)


    def _plan_all_paths(self, meetings, constraints):
        all_paths = {}
        for task_id, meeting in meetings.items():
            paths_group, success = self._plan_group(task_id, meeting, constraints)
            if not success:
                return None, False
            all_paths.update(paths_group)
            # print(task_id, meeting)
        return all_paths, True

    def _plan_group(self, task_id, meeting, constraints):
        task = self.tasks[task_id]
        alpha_id = f"alpha_{task_id}"
        beta_id = f"beta_{task_id}"
        v_m, t_m = meeting

        # 1. 提取所有 C++ 函数所需的参数

        # 任务数据
        start_alpha = task['start_alpha']
        goal_alpha = task['goal_alpha']
        start_beta = task['start_beta']
        pickup_beta = task['pickup_beta']  # s_i

        # 约束
        cons_alpha = constraints.get(alpha_id, {'vertex': set(), 'edge': set()})
        cons_beta = constraints.get(beta_id, {'vertex': set(), 'edge': set()})
        # 持续时间
        co_work_duration = task.get('co_work_duration', 0)
        pickup_duration = task.get('pickup_duration', 0)
        delivery_duration = task.get('delivery_duration', 0)

        try:
            path_alpha, path_beta = self.planner.plan_group_paths_flexible(
                start_alpha, goal_alpha,
                start_beta, pickup_beta,
                v_m, int(t_m),
                cons_alpha, cons_beta,
                co_work_duration,
                pickup_duration,
                delivery_duration
            )
        except Exception as e:
            # 捕获 C++ 侧可能发生的异常 (例如 py::cast 失败)
            print(f"!! C++ plan_group_paths 异常 (Task {task_id}): {e}")
            return None, False

        # 3. 检查 C++ 返回的空路径 (表示失败)
        #    (C++ 返回 ([], []) 时, path_alpha 将为 false)
        if not path_alpha:
            return None, False

        return {alpha_id: path_alpha, beta_id: path_beta}, True

    def _get_intentional_meetings_with_wildcard(self, meetings):
        intentional = {}
        for task_id, (v_m, t_m) in meetings.items():
            pair = {f"alpha_{task_id}", f"beta_{task_id}"}
            intentional[(v_m, -1)] = pair
        return intentional

    def _get_intentional_meetings(self, paths, meetings):
        """
        Helper to identify valid overlapping at meeting points.
        Crucial for Co-MAPF to allow agents to meet[cite: 104].
        """
        intentional = {}
        for meetingTable in self.meeting_tables.items():
            task_id = meetingTable[0]
            m_point = meetings[0][0]
            alpha_id = f"alpha_{task_id}"
            beta_id = f"beta_{task_id}"
            if alpha_id not in paths or beta_id not in paths: continue

            path_a = paths[alpha_id]
            path_b = paths[beta_id]

            # Detect overlap at m_point
            limit = min(len(path_a), len(path_b))
            for t in range(limit):
                if path_a[t] == m_point and path_b[t] == m_point:
                    intentional[(m_point, t)] = {alpha_id, beta_id}
        return intentional

