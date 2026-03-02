import os
import random
import time
import sys
import argparse
import numpy as np

from utils.mapLoader import MapLoader
from utils.mapf_utils import calculate_makespan, calculate_soc

# Import the CPP Solver
try:
    from algo.pp import CooperativePriorityPlanning
except ImportError as e:
    print(f"Error: Failed to import CooperativePriorityPlanning. Details: {e}")
    sys.exit(1)


def parse_arguments():
    """
    Defines and parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Cooperative Priority Planning (CPP) Runner")

    # --- Environment Settings ---
    parser.add_argument("--map", type=str, default="maps/random-32-32-20.map",
                        help="Path to the map file")
    # [新增] 任务文件路径，如果指定此参数，将忽略 num_tasks
    parser.add_argument("--task_file", type=str, default=None,
                        help="Path to a custom task file (.txt). Overrides --num_tasks.")
    parser.add_argument("--num_tasks", type=int, default=5,
                        help="Number of tasks to generate (only used if --task_file is not set)")
    parser.add_argument("--seed", type=int, default=150,
                        help="Random seed for reproducibility")

    # --- Solver Settings ---
    parser.add_argument("--objective", type=str, default="Makespan", choices=["Makespan", "SoC"],
                        help="Optimization objective")
    parser.add_argument("--max_time", type=int, default=800,
                        help="Maximum simulation time steps")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Solver timeout in seconds")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output")

    # --- Algorithm Specific (CPP/IPP) ---
    parser.add_argument("--heuristic", type=str, default="LFF",
                        choices=["LFF", "LPF", "Random"],
                        help="Heuristic for task prioritization")
    parser.add_argument("--flexible", action="store_true",
                        help="Enable Flexible Time (Dynamic T) mode")
    parser.add_argument("--time_window", type=int, default=10,
                        help="Time window size for Fixed T mode")

    # --- Heatmap / Congestion Settings ---
    parser.add_argument("--use_heatmap", action="store_true", default=True,
                        help="Enable congestion heatmap guidance (Default: True)")
    parser.add_argument("--no_heatmap", action="store_false", dest="use_heatmap",
                        help="Disable congestion heatmap guidance")
    parser.add_argument("--heatmap_weight", type=float, default=2.0,
                        help="Penalty weight for traversing high-traffic areas")

    return parser.parse_args()


def load_tasks_from_file(file_path):
    """
    解析自定义任务文件。
    每一行代表一个任务，格式见文档描述。
    """
    tasks = {}
    print(f"Loading tasks from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Task file not found: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过空行或注释
            if not line or line.startswith("#"):
                continue

            parts = list(map(int, line.split()))

            # 简单校验数据长度 (ID + 4个坐标点(8个数) + 3个持续时间 = 12个数)
            if len(parts) < 12:
                print(f"Warning: Skipping invalid line (not enough data): {line}")
                continue

            task_id = parts[0]
            tasks[task_id] = {
                'start_alpha': (parts[1], parts[2]),
                'goal_alpha': (parts[3], parts[4]),
                'start_beta': (parts[5], parts[6]),
                'pickup_beta': (parts[7], parts[8]),
                'co_work_duration': parts[9],
                'pickup_duration': parts[10],
                'delivery_duration': parts[11]
            }

    return tasks


def main():
    # 1. Parse Arguments
    args = parse_arguments()

    # 2. Setup Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 3. Load Map
    print(f"[{args.map}] Loading map...")
    if not os.path.exists(args.map):
        print(f"Error: Map file not found at {args.map}")
        sys.exit(1)

    parsed_grid_np, _ = MapLoader.load_map_file(args.map)

    # 4. Generate OR Load Tasks
    # [修改] 优先判断是否存在 task_file 参数
    if args.task_file:
        try:
            tasks = load_tasks_from_file(args.task_file)
            # 更新 args.num_tasks 以便后续逻辑一致
            args.num_tasks = len(tasks)
        except Exception as e:
            print(f"Error loading task file: {e}")
            sys.exit(1)
    else:
        print(f"Generating {args.num_tasks} random tasks...")
        valid_coords = MapLoader.get_largest_component(parsed_grid_np)
        tasks = MapLoader.generate_reachable_tasks(valid_coords, args.num_tasks)

    # Print Task Details
    print("\nGenerated/Loaded Tasks:")
    for task_id, data in tasks.items():
        print(f"  Task {task_id}: "
              f"Alpha({data['start_alpha']}->{data['goal_alpha']}), "
              f"Beta({data['start_beta']}->{data['pickup_beta']}) | "
              f"Co-work: {data.get('co_work_duration', 0)}")

    # 5. Initialize Solver
    print(f"\n======= Running CPP Solver =======")
    print(f"Config: Objective={args.objective}, Heatmap={args.use_heatmap} (w={args.heatmap_weight}), Heuristic={args.heuristic}")

    # Map heuristic naming if necessary
    heuristic_name = args.heuristic
    if args.heuristic == "LFF":
        heuristic_name = "Least Flexible First"
    elif args.heuristic == "LPF":
        heuristic_name = "Longest Path First"
    print(f"Heuristic Strategy: {heuristic_name}")

    print(f"Max Time: {args.max_time}, Timeout: {args.timeout} seconds")

    try:
        solver = CooperativePriorityPlanning(
            grid=parsed_grid_np,
            tasks=tasks,
            args=args
        )
    except Exception as e:
        print(f"Initialization Failed: {e}")
        sys.exit(1)

    # 6. Run Search
    start_time = time.time()
    solution_paths = solver.solve()
    runtime = time.time() - start_time

    print(f"\n======= Execution Finished in {runtime:.4f} seconds =======")

    # 7. Analyze Results
    if solution_paths:
        print(">> Solution Found!")

        makespan = calculate_makespan(solution_paths)
        soc = calculate_soc(solution_paths)
        stats = solver.get_stats()

        print("-" * 30)
        print(f"Makespan        : {makespan}")
        print(f"Sum of Costs    : {soc}")
        print("-" * 30)
        print(f"Detailed Stats  : {stats}")
        print("-" * 30)
    else:
        print(">> Failure: No solution found.")


if __name__ == '__main__':
    main()
