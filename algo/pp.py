import heapq
import numpy as np
import time
import random

from utils.heatmap_generator import generate_congestion_heatmap
from utils.mapf_utils import calculate_makespan, calculate_soc

# Import C++ library
try:
    from planner_lib import planner_lib
except ImportError:
    try:
        from planner_lib import planner_lib
    except ImportError:
        print("=" * 50)
        print("Error: Cannot import C++ core 'planner_lib'.")
        exit(1)


class MeetingsTable:
    """
    Implementation of Co-CBS Algorithm 2: Pre-computing the ideal meeting point table.
    """

    def __init__(self, planner, task_id, task_data, objective):
        self.planner = planner
        self.task_id = task_id
        self.task = task_data
        self.objective = objective
        self.heap = []  # (estimated_total_cost, v_m, t_m, base_time_cost)
        self.pickup_duration = self.task.get('pickup_duration', 0)

        # 1. Shortest path for Beta to Pickup
        path_b_pre, self.cost_b_pre = self.planner.a_star_search(self.task['start_beta'], self.task['pickup_beta'])
        if not path_b_pre:
            return

        # 2. Precompute Dijkstra maps
        self.dists_alpha_start_to_all = self.planner.run_dijkstra(self.task['start_alpha'])
        self.dists_s_to_all = self.planner.run_dijkstra(self.task['pickup_beta'])
        self.dists_goal_to_all = self.planner.run_dijkstra(self.task['goal_alpha'])

        # 3. Populate meeting table
        all_vertices = set(self.dists_alpha_start_to_all.keys()) | set(self.dists_s_to_all.keys())
        for v in all_vertices:
            if (v not in self.dists_alpha_start_to_all or
                    v not in self.dists_s_to_all or
                    v not in self.dists_goal_to_all):
                continue

            t_alpha = self.dists_alpha_start_to_all[v]
            cost_s_to_v = self.dists_s_to_all[v]

            if t_alpha == float('inf') or cost_s_to_v == float('inf'): continue

            # Beta arrival time
            t_beta = self.cost_b_pre + self.pickup_duration + cost_s_to_v

            # Meeting start time
            t_star_v = max(t_alpha, t_beta)

            cost_v_g = self.dists_goal_to_all[v]
            if cost_v_g == float('inf'): continue
            base_cost = 2 * t_star_v + cost_v_g
            heapq.heappush(self.heap, (base_cost, v, t_star_v, base_cost))

    def get_next_meeting(self, current_heatmap=None, heatmap_weight=10.0):
        while self.heap:
            stored_cost, v_m, t_m, base_cost = heapq.heappop(self.heap)

            penalty = 0.0
            if current_heatmap is not None:
                r, c = v_m
                penalty = current_heatmap[r, c] * heatmap_weight

            real_cost = base_cost + penalty

            if real_cost > stored_cost + 1e-5:
                heapq.heappush(self.heap, (real_cost, v_m, t_m, base_cost))
                continue

            new_base = base_cost + 1
            new_total = new_base + penalty
            heapq.heappush(self.heap, (new_total, v_m, t_m + 1, new_base))

            return (v_m, t_m), real_cost

        return (None, None), float('inf')


class CooperativePriorityPlanning:
    def __init__(self, grid, tasks: dict,
                 objective="Makespan", max_sim_time=800, heuristic_method="Longest Path First",
                 use_heatmap=False, debug=False, time_limit=None, time_window=10,
                 **kwargs):
        self.is_debug = debug
        self.tasks = tasks
        self.objective = objective
        self.max_sim_time = max_sim_time
        self.time_window = time_window
        self.meeting_tables = {}
        self.calculate_solution_cost = calculate_makespan if objective == "Makespan" else calculate_soc
        self.grid = grid
        self.time_limit = time_limit

        # [Statistics Initialization]
        self.stats = {
            "low_level_searches": 0,  # Number of low-level planning attempts
            "total_wait_steps": 0,  # Total number of wait steps
            "pure_search_time": 0.0,  # Pure search duration (excluding preprocessing)
            "max_congestion": 0,  # Maximum congestion
            "congestion_variance": 0.0  # Congestion variance
        }

        # Initialize C++ Planner
        self.planner = planner_lib(self.grid, max_sim_time, debug)
        self.heatmap_weight = 10.0

        self.heuristic_method = heuristic_method
        self.use_heatmap = use_heatmap

        self.heatmap = None
        if self.use_heatmap:
            # print("[PriorityPlanner] Analyzing map topology to generate congestion heatmap...")
            self.heatmap = generate_congestion_heatmap(self.grid, self.tasks)
            if not isinstance(self.heatmap, np.ndarray):
                self.heatmap = np.array(self.heatmap)
            # print("[PriorityPlanner] Heatmap generation complete.")
            pass
        else:
            # print("[PriorityPlanner] Heatmap optimization disabled (Standard Heuristic).")
            pass

        # 2. Initialize Meeting Tables
        try:
            for task_id, task_data in self.tasks.items():
                self.meeting_tables[task_id] = MeetingsTable(
                    self.planner,
                    task_id,
                    task_data,
                    self.objective
                )
        except ValueError as e:
            raise Exception(f"!! Priority planner pre-computation failed: {e}")

        # 3. Calculate heuristics and sort
        assert heuristic_method in ["Longest Path First", "Least Flexible First", "Random"]
        self.task_heuristics = {}
        for task_id, table in self.meeting_tables.items():
            if not table.heap:
                best_cost = float('inf')
                num_options = 0
            else:
                best_cost = table.heap[0][0]
                num_options = len(table.heap)

            self.task_heuristics[task_id] = {
                'best_cost': best_cost,
                'flexibility': num_options
            }

        self.priority_order = []

    def _add_path_as_constraints(self, v_cons_set, e_cons_set, path, infinite_stay=True):
        if not path: return
        max_path_time = len(path) - 1
        for t in range(max_path_time):
            v_cons_set.add((path[t], t))
            if path[t] != path[t + 1]:
                e_cons_set.add(((path[t], path[t + 1]), t + 1))
                e_cons_set.add(((path[t + 1], path[t]), t + 1))
        final_pos = path[max_path_time]
        v_cons_set.add((final_pos, max_path_time))
        if infinite_stay:
            for t in range(max_path_time + 1, self.max_sim_time + 1):
                v_cons_set.add((final_pos, t))

    def _update_heatmap_for_task(self, path_list, v_m, start_a, start_b):
        if self.heatmap is None: return
        height, width = self.heatmap.shape

        # 1. Base path heat
        for path in path_list:
            for r, c in path:
                if 0 <= r < height and 0 <= c < width:
                    self.heatmap[r, c] += 0.2  # Heat for passing through

        # 2. Meeting point high heat
        if v_m:
            vm_r, vm_c = v_m
            if 0 <= vm_r < height and 0 <= vm_c < width:
                self.heatmap[vm_r, vm_c] += 1.0

    def solve(self, infinite_stay=True):
        """
        Execute the priority planning algorithm.
        """
        start_time = time.time()
        # [Stats] Record pure search start time
        search_start_time = time.time()

        all_paths = {}
        master_vertex_constraints = set()
        master_edge_constraints = set()
        finished_goals = {}

        if self.heuristic_method == "Longest Path First":
            priority_order = sorted(self.tasks.keys(), key=lambda t: self.task_heuristics[t]['best_cost'], reverse=True)
        elif self.heuristic_method == "Least Flexible First":
            priority_order = sorted(self.tasks.keys(), key=lambda t: self.task_heuristics[t]['flexibility'])
        else:
            priority_order = list(self.tasks.keys())
            random.shuffle(priority_order)

        self.priority_order = priority_order

        for i, task_id in enumerate(priority_order):
            # [Timeout Check]
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                return None

            alpha_id = f"alpha_{task_id}"
            beta_id = f"beta_{task_id}"
            task_data = self.tasks[task_id]
            table = self.meeting_tables[task_id]
            banned_static_goals = set()

            # [Retry Loop]
            while True:
                if self.time_limit and (time.time() - start_time) > self.time_limit:
                    return None
                if self.is_debug:
                    print("Planning task:", task_id, time.time() - start_time)

                # 1. Prepare constraints
                current_constraints = {
                    alpha_id: {'vertex': master_vertex_constraints.copy(), 'edge': master_edge_constraints.copy()},
                    beta_id: {'vertex': master_vertex_constraints.copy(), 'edge': master_edge_constraints.copy()}
                }

                if self.use_heatmap and self.heatmap is not None:
                    current_constraints[alpha_id]['heatmap'] = self.heatmap
                    current_constraints[beta_id]['heatmap'] = self.heatmap

                if banned_static_goals:
                    for pos in banned_static_goals:
                        for t in range(self.max_sim_time + 1):
                            current_constraints[alpha_id]['vertex'].add((pos, t))
                            current_constraints[beta_id]['vertex'].add((pos, t))

                # 2. Attempt to find path (Iterate through meeting points)
                path_found_in_meeting_loop = False
                path_alpha, path_beta = [], []

                while True:
                    if self.time_limit and (time.time() - start_time) > self.time_limit:
                        return None

                    meeting_info = table.get_next_meeting(self.heatmap, self.heatmap_weight)
                    if meeting_info[0] is None: break

                    meeting, cost = meeting_info
                    v_m, t_m = meeting

                    if v_m in banned_static_goals: continue

                    # [Stats] Increment search count
                    self.stats["low_level_searches"] += 1

                    path_alpha, path_beta = self.planner.plan_group_paths_fixed_t(
                        task_data['start_alpha'], task_data['goal_alpha'],
                        task_data['start_beta'], task_data['pickup_beta'],
                        v_m, self.time_window,
                        current_constraints[alpha_id], current_constraints[beta_id],
                        task_data.get('co_work_duration', 0), task_data.get('pickup_duration', 0), task_data.get('delivery_duration', 0)
                    )

                    if path_alpha and path_beta:
                        path_found_in_meeting_loop = True
                        break

                if not path_found_in_meeting_loop:
                    print(f"!! Solver failed: Task {task_id} has no solution under current constraints.")
                    return None

                # 3. [Clipping/Collision check]
                collision_detected = False
                # ... (Keep original collision check logic if uncommented) ...

                if collision_detected:
                    continue
                else:
                    # 4. Success
                    all_paths[alpha_id] = path_alpha
                    all_paths[beta_id] = path_beta

                    # [Stats] Calculate wait steps
                    for path in [path_alpha, path_beta]:
                        if not path: continue
                        waits = sum(1 for t in range(1, len(path)) if path[t] == path[t - 1])
                        self.stats["total_wait_steps"] += waits

                    self._add_path_as_constraints(master_vertex_constraints, master_edge_constraints, path_alpha, infinite_stay)
                    self._add_path_as_constraints(master_vertex_constraints, master_edge_constraints, path_beta, infinite_stay)

                    finished_goals[path_alpha[-1]] = len(path_alpha) - 1
                    finished_goals[path_beta[-1]] = len(path_beta) - 1

                    if self.use_heatmap:
                        self._update_heatmap_for_task([path_alpha, path_beta], v_m, task_data['start_alpha'], task_data['start_beta'])
                    break

        # [Stats] Record pure search time
        self.stats["pure_search_time"] = time.time() - search_start_time

        # [Stats] Calculate congestion distribution metrics
        if all_paths:
            usage_counts = {}
            for agent_path in all_paths.values():
                for loc in agent_path:
                    usage_counts[loc] = usage_counts.get(loc, 0) + 1
            if usage_counts:
                vals = list(usage_counts.values())
                self.stats["max_congestion"] = np.max(vals)
                self.stats["congestion_variance"] = np.var(vals)
        # print(self.stats)
        return all_paths

    def get_stats(self):
        """ Return statistics dictionary """
        return self.stats