def validate_final_path(path_alpha, path_beta, task_config, task_id="Unknown"):
    """
    [组内检查] 验证单个任务组的 Alpha/Beta 协作逻辑。
    检查项: 移动连续性、唯一汇合、刚性时长达标。
    修改点：
    1. 移除了 Alpha 投递的“隐式无限停留”假设，要求显式等待步数。
    """
    msgs = []
    is_valid = True

    if not path_alpha or not path_beta:
        return False, ["路径为空"]

    len_a = len(path_alpha)
    len_b = len(path_beta)
    max_len = max(len_a, len_b)

    def get_pos(path, t):
        return path[t] if t < len(path) else None  # [修改] 越界返回 None

    # 1. 识别汇合 (重叠点)
    overlap_indices = []
    for t in range(max_len):
        pos_a = get_pos(path_alpha, t)
        pos_b = get_pos(path_beta, t)
        if pos_a is not None and pos_b is not None and pos_a == pos_b:
            overlap_indices.append(t)

    if not overlap_indices:
        return False, ["Alpha 和 Beta 全程未汇合"]

    # 2. 验证“唯一汇合” (Single Continuous Rendezvous)
    first_meet_t = overlap_indices[0]
    last_meet_t = overlap_indices[-1]
    expected_len = last_meet_t - first_meet_t + 1

    # 如果实际重叠点数量 != 理论连续长度，说明中间断开了 -> 溜溜球行为
    if len(overlap_indices) != expected_len:
        is_valid = False
        msgs.append(
            f"非法多次汇合 (Yo-Yo): t={first_meet_t} 碰面, t={last_meet_t} 结束, 但中间有 {expected_len - len(overlap_indices)} 个时刻分开了。")

    # 3. 验证协作时长
    req_cowork = task_config.get('co_work_duration', 0)
    if len(overlap_indices) < req_cowork:
        is_valid = False
        msgs.append(f"协作时长不足: 需 {req_cowork}, 实 {len(overlap_indices)}")

    # 4. 验证 Beta 取货 (Pickup Duration)
    req_pickup = task_config.get('pickup_duration', 0)
    pickup_loc = task_config.get('pickup_beta')

    if req_pickup > 0 and pickup_loc:
        found_valid_pickup = False
        current_streak = 0
        for pos in path_beta:
            if pos == pickup_loc:
                current_streak += 1
                if current_streak >= req_pickup:
                    found_valid_pickup = True
                    break
            else:
                current_streak = 0

        if not found_valid_pickup:
            is_valid = False
            msgs.append(f"Beta 取货等待不足: 未发现连续 {req_pickup} 步停留在 {pickup_loc}")

    # 5. 验证 Alpha 投递 (Delivery Duration)
    # [修改] 严格模式：必须在路径中包含显式的等待步数
    req_delivery = task_config.get('delivery_duration', 0)
    goal_loc = task_config.get('goal_alpha')

    if req_delivery > 0 and goal_loc:
        found_valid_delivery = False
        current_streak = 0

        # 投递必须发生在汇合之后
        start_search_idx = last_meet_t + 1

        # 在新逻辑下，Alpha 的路径应该正好在完成投递后结束（或稍后）
        # 我们扫描路径末尾段
        for t in range(start_search_idx, len_a):
            if path_alpha[t] == goal_loc:
                current_streak += 1
                if current_streak >= req_delivery:
                    found_valid_delivery = True
                    break
            else:
                current_streak = 0

        if not found_valid_delivery:
            is_valid = True
            msgs.append(f"Alpha 投递等待不足: 汇合后未在终点连续停留 {req_delivery} 步 (Disappear at Target 模式要求显式等待)")

    return is_valid, msgs


def validate_scenario(all_paths, tasks):
    """
    [全局检查] 检查所有智能体之间的冲突 (包括不同组)。
    修改点：
    1. 实现了 Disappear at Target 逻辑：一旦 t >= len(path)，智能体视为消失，不产生冲突。
    """
    # print("\n" + "=" * 50)
    # print("🛡️  全局路径合法性校验 (Global Validator)")
    # print("=" * 50)

    error_count = 0

    # 1. 组内逻辑检查
    for task_id, config in tasks.items():
        alpha_id = f"alpha_{task_id}"
        beta_id = f"beta_{task_id}"

        if alpha_id not in all_paths or beta_id not in all_paths:
            continue

        ok, msgs = validate_final_path(all_paths[alpha_id], all_paths[beta_id], config, task_id)
        if not ok:
            error_count += 1
            print(f"❌ [Task {task_id}] 逻辑错误: {'; '.join(msgs)}")

    # 2. 全局冲突检查 (O(N^2 * T))
    agents = list(all_paths.keys())

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            name1 = agents[i]
            name2 = agents[j]
            path1 = all_paths[name1]
            path2 = all_paths[name2]

            # 判断是否为同一任务组 (同一组在汇合期间允许顶点重叠)
            try:
                task_id1 = name1.split('_')[1]
                task_id2 = name2.split('_')[1]
                is_same_group = (task_id1 == task_id2)
            except IndexError:
                is_same_group = False

            len1 = len(path1)
            len2 = len(path2)
            max_t = max(len1, len2)

            # [关键] 获取位置函数：超出长度返回 None
            def get_pos_at_t(path, length, t):
                return path[t] if t < length else None

            for t in range(max_t):
                pos1 = get_pos_at_t(path1, len1, t)
                pos2 = get_pos_at_t(path2, len2, t)

                # 如果任意一方已经消失，则不可能发生冲突
                if pos1 is None or pos2 is None:
                    continue

                # (A) 顶点冲突
                if pos1 == pos2:
                    # 如果不是同一组，绝对冲突
                    # 如果是同一组，通常允许汇合，所以忽略 (更严格的检查在 validate_final_path 做)
                    if not is_same_group:
                        print(f"💥 [冲突] {name1} 与 {name2} 在 t={t} 发生顶点重叠 {pos1}")
                        error_count += 1

                # (B) 边冲突 (对穿) - 不允许任何人对穿
                if t > 0:
                    prev1 = get_pos_at_t(path1, len1, t - 1)
                    prev2 = get_pos_at_t(path2, len2, t - 1)

                    # 双方在 t-1 和 t 时刻都必须存在，才能发生交换冲突
                    if prev1 is not None and prev2 is not None:
                        # 1: u->v, 2: v->u
                        if prev1 == pos2 and prev2 == pos1:
                            # 排除原地不动的情况 (实际上如果 pos1!=prev1 且发生了交换，就是冲突)
                            if prev1 != pos1:
                                print(f"⚔️ [对穿] {name1} 与 {name2} 在 t={t - 1}->{t} 发生边冲突!")
                                print(f"   {name1}: {prev1}->{pos1}")
                                print(f"   {name2}: {prev2}->{pos2}")
                                error_count += 1

    if error_count == 0:
        # print("\n✨✨ 完美！所有路径均合法且无冲突。 ✨✨")
        return True
    else:
        # print(f"\n🚫 校验完成，共发现 {error_count} 个问题。请检查代码逻辑。")
        return False