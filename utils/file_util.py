import os
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(marker: str = ".venv") -> Path:
    """
    从当前脚本文件位置开始，向上查找并返回项目根目录。

    这个函数通过寻找一个“标记”文件或目录（如`.git`）来确定根目录。
    无论当前脚本在根目录、或在任意深度的子目录，都能正确工作。

    :param marker: 用来标识项目根目录的文件或文件夹名。
                   常用标记包括: ".git", "pyproject.toml", "requirements.txt",
                   或自定义的空文件如 ".project_root"。
    :return: 项目根目录的 Path 对象。
    :raises FileNotFoundError: 如果向上查找到文件系统根部仍未找到标记。
    """
    # 从当前文件 (__file__) 的目录开始搜索
    # .resolve() 用于获取绝对路径，以处理符号链接等情况
    current_dir = Path(__file__).parent.resolve()

    while True:
        # 1. 检查当前目录是否包含标记
        if (current_dir / marker).exists():
            return current_dir

        # 2. 检查是否已到达文件系统的根目录
        # 当一个目录的父目录是它自身时，说明已达顶层（如 '/' 或 'C:\'）
        if current_dir == current_dir.parent:
            # 在根目录最后检查一次
            if (current_dir / marker).exists():
                return current_dir
            # 如果到达顶层仍未找到，则抛出异常
            raise FileNotFoundError(
                f"无法找到项目根目录：向上搜索至文件系统根部仍未找到标记 '{marker}'。"
            )

        # 3. 如果未找到且未到顶层，则向上移动一级
        current_dir = current_dir.parent


def parse_grid_from_str(grid_str):
    """
    从多行字符串解析网格。
    字符为 '.' 表示可通行，其余一律视为障碍。
    """
    map_rows_raw = grid_str.strip().split('\n')
    map_rows_cleaned = []
    for row in map_rows_raw:
        cleaned_line = row.strip()
        if cleaned_line:
            # 只取第一个空格前的部分（保留你原来的逻辑）
            map_data = cleaned_line.split(' ')[0]
            map_rows_cleaned.append(map_data)

    grid = np.array([
        [0 if ch == '.' else 1 for ch in row]
        for row in map_rows_cleaned
    ])
    return grid


def custom_deep_copy_constraints(constraints):
    """
    [创新点] 这是一个 copy.deepcopy 的高性能替代品。
    它专门用于拷贝 agent_constraints 的特定结构：
    { agent_id: {'vertex': set(), 'edge': set()} }
    """
    new_constraints = {}
    for agent_id, inner_dict in constraints.items():
        # 创建第二层字典
        # 拷贝 'vertex' 和 'edge' 键对应的 *集合*
        new_constraints[agent_id] = {
            'vertex': inner_dict['vertex'].copy(),
            'edge': inner_dict['edge'].copy()
        }
    return new_constraints


# ==========================================
# 3. 后处理：计算均值并重写 CSV
# ==========================================
def finalize_results_with_summary(csv_file):
    if not os.path.exists(csv_file): return
    print(f"\n正在处理最终结果: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
        if df.empty: return

        # 生成 Excel
        xlsx_name = csv_file.replace(".csv", ".xlsx")
        try:
            pivot_success = df.pivot_table(index='Num_Tasks', columns='Algorithm', values='Success', aggfunc='mean')
            df_success = df[df['Success'] == 1]
            pivot_runtime = df_success.pivot_table(index='Num_Tasks', columns='Algorithm', values='Runtime', aggfunc='mean')
            pivot_mksp = df_success.pivot_table(index='Num_Tasks', columns='Algorithm', values='Makespan', aggfunc='mean')

            # [新增] 额外指标的透视表
            pivot_search = df_success.pivot_table(index='Num_Tasks', columns='Algorithm', values='LowLevelSearches', aggfunc='mean')
            pivot_wait = df_success.pivot_table(index='Num_Tasks', columns='Algorithm', values='WaitSteps', aggfunc='mean')

            with pd.ExcelWriter(xlsx_name) as writer:
                df.to_excel(writer, sheet_name="Raw_Data", index=False)
                pivot_success.to_excel(writer, sheet_name="Success_Rate")
                pivot_runtime.to_excel(writer, sheet_name="Avg_Runtime")
                if not pivot_mksp.empty: pivot_mksp.to_excel(writer, sheet_name="Avg_Makespan")

                # [新增] 写入额外指标
                if not pivot_search.empty: pivot_search.to_excel(writer, sheet_name="Avg_Searches")
                if not pivot_wait.empty: pivot_wait.to_excel(writer, sheet_name="Avg_WaitSteps")

            print(f"✅ Excel 报表已生成: {xlsx_name}")
        except Exception as e:
            print(f"⚠️ Excel 生成警告: {e}")

        # 计算 Summary
        group_cols = ['Algorithm', 'Objective'] if 'Objective' in df.columns else ['Algorithm']
        summary = df.groupby(group_cols).agg(
            Total_Instances=('Success', 'count'),
            Success_Rate=('Success', 'mean'),
            Avg_Runtime=('Runtime', 'mean')
        )
        if not df_success.empty:
            # [新增] 聚合更多指标
            metrics = df_success.groupby(group_cols).agg(
                Avg_Makespan=('Makespan', 'mean'),
                Avg_SoC=('SoC', 'mean'),
                Avg_SearchTime=('SearchTime', 'mean'),
                Avg_Searches=('LowLevelSearches', 'mean'),
                Avg_WaitSteps=('WaitSteps', 'mean'),
                Avg_CongestionVar=('CongestionVar', 'mean')
            )
            summary = summary.join(metrics)
        summary = summary.round(4)

        # 重写 CSV
        csv_content = []
        csv_content.append("=== Summary Statistics (Averaged) ===")
        csv_content.append(summary.to_csv())
        csv_content.append("\n")
        csv_content.append("=== Raw Instance Results ===")
        csv_content.append(df.to_csv(index=False))

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            f.write("\n".join(csv_content))

    except Exception as e:
        print(f"❌ 后处理失败: {e}")

