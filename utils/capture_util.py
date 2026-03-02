import pickle
import os
import time
import numpy as np

def save_failure_case(folder_name, case_prefix, grid_np, start_alpha, goal_alpha, start_beta, pickup_beta,
                      v_m, t_m, cons_alpha, cons_beta, durations):
    """
    保存低层规划失败的现场快照。
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 生成唯一文件名
    filename = f"{folder_name}/{case_prefix}_FAIL_{timestamp}.pkl"

    # 打包所有必要的数据
    debug_context = {
        "grid": grid_np,
        "params": {
            "start_alpha": start_alpha,
            "goal_alpha": goal_alpha,
            "start_beta": start_beta,
            "pickup_beta": pickup_beta,
            "v_m": v_m,
            "t_m": t_m,
            "co_work_duration": durations[0],
            "pickup_duration": durations[1],
            "delivery_duration": durations[2]
        },
        "constraints": {
            "alpha": cons_alpha,
            "beta": cons_beta
        }
    }

    try:
        with open(filename, "wb") as f:
            pickle.dump(debug_context, f)
        print(f"\n[DEBUG] 规划失败现场已保存至: {filename}")
    except Exception as e:
        print(f"\n[DEBUG] 保存失败现场时出错: {e}")