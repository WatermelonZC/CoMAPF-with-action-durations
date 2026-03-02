import heapq
import copy
import time
import numpy as np

try:
    from ecocbs_cpp_lib import EcoCBS_Planner
except ImportError:
    try:
        from algo.ecocbs_cpp_lib import EcoCBS_Planner
    except ImportError:
        print("=" * 50)
        print("错误: 无法导入 C++ 核心 'ecocbs_cpp_lib'。")
        exit(1)

from env.ctnode import CTNode
from utils.file_util import custom_deep_copy_constraints
from utils.mapf_utils import calculate_makespan, calculate_soc, _find_conflict


class CBTMP_Solver:
    """
    CBTMP Solver (Algorithm 1 in Paper [cite: 118])
    Structure:
    1. High-Level Policy (Meeting Point Selection) - Algorithm 2 [cite: 183]
    2. Low-Level Policy (Path Planning with Time Alignment) - Algorithm 4 [cite: 335]
    3. Conflict Resolution via CBS - Algorithm 3 [cite: 271]
    """

    # [修改 1] 在 __init__ 中增加 time_limit 参数
    def __init__(self, grid, tasks, max_sim_time=1000, objective="Makespan", debug=False, time_limit=None):
        """
        :param objective: "Makespan" or "SumOfCosts" (Determines Eq. 1 or Eq. 4 in paper)
        :param time_limit: 最大允许运行时间（秒），超过则返回失败
        """
        self.grid = grid
        self.tasks = tasks
        self.objective = objective  # [cite: 127] Paper evaluates both
        self.planner = EcoCBS_Planner(self.grid, max_sim_time, debug)
        self.meeting_points = {}
        self.time_limit = time_limit  # [新增] 保存时间限制

        # Select cost calculation function based on paper definition
        if self.objective == "Makespan":
            self.calc_sol_cost = calculate_makespan
        else:
            self.calc_sol_cost = calculate_soc

    def solve(self):
        start_time = time.time()

        # Step 1: Calculate Meeting Points
        for task_id, task in self.tasks.items():
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                # print(f"!! Solver Failed: Timeout during Meeting Point Calculation")
                return None
            m_point, cost_val = self._find_best_meeting_point(task)
            if m_point is None:
                return None
            self.meeting_points[task_id] = m_point

        # Step 2: Generate Root Node
        root_paths = {}
        root_constraints = {}

        for task_id in self.tasks:
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                return None
            root_constraints[f"alpha_{task_id}"] = {'vertex': set(), 'edge': set()}
            root_constraints[f"beta_{task_id}"] = {'vertex': set(), 'edge': set()}
            paths, _, success = self._plan_group(task_id, root_constraints)
            if not success:
                return None
            root_paths.update(paths)

        root_cost = self.calc_sol_cost(root_paths)

        # [新增] 计算 Root 的冲突数量
        root_conflicts = self._compute_conflict_count(root_paths)

        # [修改] 创建 Root 节点时传入 conflict_count
        root_node = CTNode(root_constraints, root_paths, root_cost, conflict_count=root_conflicts)

        open_set = []
        heapq.heappush(open_set, root_node)

        # Step 3: Conflict-Based Search
        iteration = 0
        while open_set:
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                # print(f"!! Solver Failed: Timeout")
                return None

            iteration += 1
            P = heapq.heappop(open_set)

            # 获取 Intentional Meetings (用于 _find_conflict)
            intentional_meetings = self._get_intentional_meetings(P.paths)

            # 这里的 _find_conflict 只找 1 个冲突用于分支，为了性能
            conflict = _find_conflict(self.planner, P.paths, intentional_meetings, {})

            if not conflict:
                return P.paths

            # [Log] 可以打印出来看看冲突数是否在下降
            if iteration % 100 == 0:
                # print(f"Iter {iteration}: Cost {P.cost}, Conflicts {P.conflict_count}, OpenSet {len(open_set)}")
                pass
            # Branching
            for agent_to_constrain in conflict['agents']:
                new_constraints = custom_deep_copy_constraints(P.agent_constraints)

                if conflict['type'] == 'vertex':
                    new_constraints[agent_to_constrain]['vertex'].add((conflict['loc'], conflict['time']))
                else:
                    u, v = conflict['loc1'], conflict['loc2']
                    if agent_to_constrain == conflict['agents'][0]:
                        new_constraints[agent_to_constrain]['edge'].add(((u, v), conflict['time']))
                    else:
                        new_constraints[agent_to_constrain]['edge'].add(((v, u), conflict['time']))

                _, task_id_str = agent_to_constrain.split('_')
                task_id = int(task_id_str) if isinstance(list(self.tasks.keys())[0], int) else task_id_str

                new_group_paths, _, success = self._plan_group(task_id, new_constraints)

                if success:
                    new_paths = P.paths.copy()
                    new_paths.update(new_group_paths)
                    new_cost = self.calc_sol_cost(new_paths)

                    # [新增] 计算新节点的冲突数量
                    new_conflicts = self._compute_conflict_count(new_paths)

                    # [修改] 创建新节点，传入 new_conflicts
                    new_node = CTNode(new_constraints, new_paths, new_cost, conflict_count=new_conflicts)
                    heapq.heappush(open_set, new_node)

        # print("--- Solver Failed: OpenSet Empty ---")
        return None


    def _find_best_meeting_point(self, task):
        # ... (代码保持不变)
        # 1. Pre-calc Beta Leg 1 (Start -> Pickup)
        # This corresponds to l_{p,i} in Eq. 10 [cite: 263]
        path_b_pre, cost_b_pre = self.planner.a_star_search(task['start_beta'], task['pickup_beta'])
        if not path_b_pre: return None, float('inf')

        # 2. Generate Distance Maps
        map_alpha = self.planner.run_dijkstra(task['start_alpha'])  # d(V0(alpha), v)
        map_beta_pickup = self.planner.run_dijkstra(task['pickup_beta'])  # d(s, v)
        map_goal = self.planner.run_dijkstra(task['goal_alpha'])  # d(v, g) -> assumes undirected graph

        best_m = None
        min_cost = float('inf')

        # Additional durations (User requested extension)
        co_work = task.get('co_work_duration', 0)
        pickup_dur = task.get('pickup_duration', 0)

        common_nodes = set(map_alpha.keys()) & set(map_beta_pickup.keys()) & set(map_goal.keys())

        for m in common_nodes:
            d_alpha_m = map_alpha[m]
            d_pickup_m = map_beta_pickup[m]
            d_m_goal = map_goal[m]

            # Calculate Arrival Times
            t_alpha_arrive = d_alpha_m
            t_beta_arrive = cost_b_pre + pickup_dur + d_pickup_m

            # Meeting starts when both arrive
            t_meeting_start = max(t_alpha_arrive, t_beta_arrive)
            # Meeting ends after co-work
            t_meeting_end = t_meeting_start + co_work
            # Alpha reaches goal
            t_goal_reach = t_meeting_end + d_m_goal

            # --- Paper Formulas Implementation ---
            if self.objective == "Makespan":
                # Eq. 2/4[cite: 233, 240]: Max time to complete tasks
                # Makespan = max(|pi_alpha|, |pi_beta|)
                # Since Alpha continues to Goal, Alpha's time is usually the bottleneck for the Task
                current_cost = t_goal_reach

            else:  # "Makespan"
                # Eq. 1[cite: 226]: SOC = d(v, m) + d(start_others, m) ...
                current_cost = t_goal_reach + 2 * t_meeting_end

            if current_cost < min_cost:
                min_cost = current_cost
                best_m = m

        return best_m, min_cost

    def _plan_group(self, task_id, constraints):
        """
        Paper Algorithm 4: Planning Paths [cite: 335]
        Delegated to C++ EcoCBS_Planner for efficiency.
        Includes 'Time-step alignment' (waiting) logic.
        """
        task = self.tasks[task_id]
        alpha_id = f"alpha_{task_id}"
        beta_id = f"beta_{task_id}"
        m_point = self.meeting_points[task_id]

        # Heuristic hint for meeting time (optional optimization)
        t_m_hint = 10

        # Call C++ Core
        path_alpha, path_beta = self.planner.plan_group_paths_fixed_t(
            task['start_alpha'], task['goal_alpha'],
            task['start_beta'], task['pickup_beta'],
            m_point, t_m_hint,
            constraints.get(alpha_id, {'vertex': set(), 'edge': set()}),
            constraints.get(beta_id, {'vertex': set(), 'edge': set()}),
            task.get('co_work_duration', 0),
            task.get('pickup_duration', 0),
            task.get('delivery_duration', 0)
        )

        if not path_alpha or not path_beta:
            return None, None, False

        return {alpha_id: path_alpha, beta_id: path_beta}, m_point, True

    def _get_intentional_meetings(self, paths):
        """
        Helper to identify valid overlapping at meeting points.
        Crucial for Co-MAPF to allow agents to meet[cite: 104].
        """
        intentional = {}
        for task_id, m_point in self.meeting_points.items():
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

    # [新增] 辅助函数：计算当前路径的所有冲突数量
    def _compute_conflict_count(self, paths):
        # 必须考虑 Intentional Meetings，否则会将合法的汇合算作冲突
        intentional_meetings = self._get_intentional_meetings(paths)
        # 调用 C++ 的 count_all_conflicts (假设上一轮我们已经绑定了这个函数)
        # 如果 C++ 没暴露这个函数，这里会报错，需要确保 C++ EcoCBS_Planner 有这个方法
        return self.planner.count_all_conflicts(paths, intentional_meetings, {})
