import itertools


class CTNode:
    """
    冲突树 (Conflict Tree) 节点。
    [修改] 增加了 conflict_count 和 id 用于更高级的 Tie-breaking
    """
    # 全局计数器，确保每个节点由唯一的 ID，保证比较的稳定性
    _id_counter = itertools.count()

    def __init__(self, agent_constraints, paths, cost, conflict_count=0):
        self.agent_constraints = agent_constraints
        self.paths = paths
        self.cost = cost
        self.conflict_count = conflict_count  # [新增] 冲突数量
        self.id = next(CTNode._id_counter)  # [新增] 唯一ID

    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        if self.conflict_count != other.conflict_count:
            return self.conflict_count < other.conflict_count

        # 如果都相同，按创建顺序（FIFO）
        return self.id < other.id


# ----------------------------------------------------------------------
# Co-CBS 节点 (CTNode)
# ----------------------------------------------------------------------
class Co_CBS_CTNode:
    """
    Co-CBS 冲突树 (CT) 节点。
    [已修改] 增加了 conflict_count (冲突计数) 用于启发式排序。
    """

    def __init__(self, agent_constraints, paths, cost, meetings, is_root=False, conflict_count=0):
        """
        初始化一个 CT 节点。

        Args:
            agent_constraints (dict): 此节点的约束。
            paths (dict): 此节点的路径解。
            cost (float): 此节点解的成本 (Makespan 或 SOC)。
            meetings (dict): 此节点的汇合点 (task_id -> (v_m, t_m))。
            is_root (bool): 是否为根节点 (用于 Algorithm 3)。
            conflict_count (int): [新增] 此节点解中的总冲突数。
        """
        self.agent_constraints = agent_constraints
        self.paths = paths
        self.cost = cost
        self.meetings = meetings
        self.is_root = is_root
        self.conflict_count = conflict_count

        # 用于 heapq 在成本和冲突数均相同时的稳定排序
        self.id = id(self)

    def __lt__(self, other):
        """
        比较函数，用于 heapq 排序。
        优先顺序:
        1. 按总成本 (cost) 升序。
        2. (成本相同) 按冲突计数 (conflict_count) 升序。
        3. (两者相同) 按节点 ID 保持稳定性。
        """
        if self.cost != other.cost:
            return self.cost < other.cost
        if self.conflict_count != other.conflict_count:
            return self.conflict_count < other.conflict_count
        return self.id < other.id

    def __str__(self):
        # 打印节点信息时包含冲突计数
        return f"CTNode(Cost={self.cost}, Conflicts={self.conflict_count}, Root={self.is_root})"
