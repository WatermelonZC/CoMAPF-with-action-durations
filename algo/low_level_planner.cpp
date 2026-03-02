/*
 * ECoCBS_cpp.cpp
 * 核心功能:
 * 1. 快速的时空 A* 搜索 (Space-Time A*)
 * 2. 智能冲突检测 (支持单次连续汇合豁免)
 * 3. 灵活的组路径规划 (支持自动时间对齐，去除了无限期停留检查)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono> // 新增

namespace py = pybind11;
using namespace pybind11::literals;

using Pos = std::tuple<int, int>;
using State = std::tuple<int, int, int>; // row, col, time
using Path = std::vector<Pos>;
using PathMap = std::unordered_map<std::string, Path>;
using VertexConstraint = std::tuple<Pos, int>;
using EdgeConstraint = std::tuple<std::tuple<Pos, Pos>, int>;

// --- Hashers ---
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct StateHasher {
    std::size_t operator()(const State& s) const {
        std::size_t seed = 0;
        hash_combine(seed, std::get<0>(s));
        hash_combine(seed, std::get<1>(s));
        hash_combine(seed, std::get<2>(s));
        return seed;
    }
};

struct PosHasher {
    std::size_t operator()(const Pos& p) const {
        std::size_t seed = 0;
        hash_combine(seed, std::get<0>(p));
        hash_combine(seed, std::get<1>(p));
        return seed;
    }
};

struct VertexConstraintHasher {
    std::size_t operator()(const VertexConstraint& vc) const {
        std::size_t seed = 0;
        const auto& pos = std::get<0>(vc);
        hash_combine(seed, std::get<0>(pos));
        hash_combine(seed, std::get<1>(pos));
        hash_combine(seed, std::get<1>(vc));
        return seed;
    }
};

struct EdgeConstraintHasher {
    std::size_t operator()(const EdgeConstraint& ec) const {
        std::size_t seed = 0;
        const auto& edge = std::get<0>(ec);
        const auto& pos1 = std::get<0>(edge);
        const auto& pos2 = std::get<1>(edge);
        hash_combine(seed, std::get<0>(pos1));
        hash_combine(seed, std::get<1>(pos1));
        hash_combine(seed, std::get<0>(pos2));
        hash_combine(seed, std::get<1>(pos2));
        hash_combine(seed, std::get<1>(ec));
        return seed;
    }
};

struct GScoreMap : public std::unordered_map<State, double, StateHasher> {
    double get(const State& key, double default_val) const {
        auto it = this->find(key);
        return (it == this->end()) ? default_val : it->second;
    }
};

// 约束结构体
struct Constraints {
    std::unordered_set<VertexConstraint, VertexConstraintHasher> vertex_cons;
    std::unordered_set<EdgeConstraint, EdgeConstraintHasher> edge_cons;
        bool has_heatmap = false;
    std::vector<std::vector<double>> heatmap;
};

using DistMap = std::unordered_map<Pos, int, PosHasher>;

struct ScopedTimer {
    std::string name;
    bool debug;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    ScopedTimer(std::string name, bool debug) : name(name), debug(debug) {
        if (debug) {
            std::cout << "[DEBUG][" << name << "] Start planning..." << std::endl;
        }
        start = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimer() {
        if (debug) {
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "[DEBUG][" << name << "] Finished. Cost: " << elapsed << " ms" << std::endl;
        }
    }
};


class planner_lib {
public:
    planner_lib(py::array_t<int> grid_np, int default_max_time = 800, bool debug = false)
      : default_max_time_(default_max_time), debug_(debug) {
        py::buffer_info buf = grid_np.request();
        if (buf.ndim != 2) throw std::runtime_error("Numpy array must be 2D");
        height_ = buf.shape[0];
        width_ = buf.shape[1];
        grid_.resize(height_, std::vector<int>(width_));
        int* ptr = static_cast<int*>(buf.ptr);
        for (py::ssize_t r = 0; r < height_; ++r) {
            for (py::ssize_t c = 0; c < width_; ++c) {
                grid_[r][c] = ptr[r * width_ + c];
            }
        }
    }

    // 解析 Python 字典为 C++ 结构体
    Constraints parse_constraints_to_cpp(const py::dict& py_cons) {
        Constraints c;
        if (py_cons.contains("vertex")) {
            c.vertex_cons = py_cons["vertex"].cast<std::unordered_set<VertexConstraint, VertexConstraintHasher>>();
        }
        if (py_cons.contains("edge")) {
            c.edge_cons = py_cons["edge"].cast<std::unordered_set<EdgeConstraint, EdgeConstraintHasher>>();
        }
        if (py_cons.contains("heatmap")) {
            c.has_heatmap = true;
            // 这里利用 pybind11 自动转换 list/numpy 到 vector<vector<double>>
            try {
                c.heatmap = py_cons["heatmap"].cast<std::vector<std::vector<double>>>();
            } catch (const std::exception& e) {
                // 防止传入 None 或格式不对导致崩溃
                c.has_heatmap = false; 
                std::cerr << "[Warn] Failed to parse heatmap: " << e.what() << std::endl;
            }
        }
        return c;
    }

    // ---------------------------------------------------------
    // 1. 核心功能：冲突检测 (支持单次连续汇合豁免)
    // ---------------------------------------------------------
    std::string get_pair_key(const std::string& a, const std::string& b) {
        return (a < b) ? (a + "+" + b) : (b + "+" + a);
    }

    py::dict find_conflict(const PathMap& paths, py::dict intentional_meetings_py, py::dict /*unused*/) {
        using IntentionalMap = std::unordered_map<VertexConstraint, std::unordered_set<std::string>, VertexConstraintHasher>;
        auto intentional_meetings = intentional_meetings_py.cast<IntentionalMap>();

        py::ssize_t max_len = 0;
        if (paths.empty()) return py::dict();
        for (const auto& pair : paths) max_len = std::max(max_len, (py::ssize_t)pair.second.size());
        if (max_len == 0) return py::dict();

        // Timeline
        std::vector<std::unordered_map<Pos, std::vector<std::string>, PosHasher>> timeline(max_len);

        for (const auto& pair : paths) {
            const auto& agent_id = pair.first; const auto& path = pair.second;
            for (py::ssize_t t = 0; t < max_len; ++t) {
                if (t >= (py::ssize_t)path.size()) continue;
                timeline[t][path[t]].push_back(agent_id);
            }
        }

        std::unordered_map<std::string, int> pair_loc_last_seen;

        // Vertex Conflict
        for (int t = 0; t < max_len; ++t) {
            for (const auto& [pos, agents] : timeline[t]) {
                if (agents.size() > 1) {
                    for (size_t i = 0; i < agents.size(); ++i) {
                        for (size_t j = i + 1; j < agents.size(); ++j) {
                            std::string a1 = agents[i];
                            std::string a2 = agents[j];
                            
                            bool is_intentional_loc = false;
                            
                            if (intentional_meetings.count({pos, t})) {
                                const auto& allowed = intentional_meetings.at({pos, t});
                                if (allowed.count(a1) && allowed.count(a2)) is_intentional_loc = true;
                            }
                            if (!is_intentional_loc && intentional_meetings.count({pos, -1})) {
                                const auto& allowed = intentional_meetings.at({pos, -1});
                                if (allowed.count(a1) && allowed.count(a2)) is_intentional_loc = true;
                            }

                            if (is_intentional_loc) {
                                size_t pos_hash = PosHasher()(pos);
                                std::string key = get_pair_key(a1, a2) + "_" + std::to_string(pos_hash);

                                if (pair_loc_last_seen.find(key) == pair_loc_last_seen.end()) {
                                    pair_loc_last_seen[key] = t;
                                } else {
                                    int last_t = pair_loc_last_seen[key];
                                    if (last_t == t - 1) {
                                        pair_loc_last_seen[key] = t;
                                    } else {
                                        return py::dict("type"_a="vertex", "time"_a=t, "loc"_a=pos, "agents"_a=agents);
                                    }
                                }
                            } else {
                                return py::dict("type"_a="vertex", "time"_a=t, "loc"_a=pos, "agents"_a=agents);
                            }
                        }
                    }
                }
            }
        }

        // Edge Conflict
        for (auto it1 = paths.begin(); it1 != paths.end(); ++it1) {
            for (auto it2 = paths.begin(); it2 != paths.end(); ++it2) {
                if (it1->first >= it2->first) continue;
                const auto& path1 = it1->second; const auto& path2 = it2->second;
                for (py::ssize_t t = 0; t < max_len - 1; ++t) {
                    if (t + 1 >= (int)path1.size() || t + 1 >= (int)path2.size()) continue;
                    const Pos& u = path1[t]; const Pos& v = path1[t+1];
                    const Pos& x = path2[t]; const Pos& y = path2[t+1];
                    if (v == x && u == y && u != x) {
                        return py::dict("type"_a="edge", "time"_a=t+1, "loc1"_a=u, "loc2"_a=v, "agents"_a=py::make_tuple(it1->first, it2->first));
                    }
                }
            }
        }
        return py::dict();
    }


    // ---------------------------------------------------------
    // 2. 新功能：基于地点的冲突检测 (支持单次连续汇合豁免)
    //    输入格式: {(x, y): {'agent1', 'agent2'}, ...}
    // ---------------------------------------------------------

    py::dict find_conflict_spatial(const PathMap& paths, py::dict intentional_locs_py) {
    // 1. 类型转换：注意这里 Key 变成了 Pos，不再是 VertexConstraint
    using IntentionalLocMap = std::unordered_map<Pos, std::unordered_set<std::string>, PosHasher>;
    auto intentional_locs = intentional_locs_py.cast<IntentionalLocMap>();

    py::ssize_t max_len = 0;
    if (paths.empty()) return py::dict();
    for (const auto& pair : paths) max_len = std::max(max_len, (py::ssize_t)pair.second.size());
    if (max_len == 0) return py::dict();

    // Timeline 构建 (完全一致)
    std::vector<std::unordered_map<Pos, std::vector<std::string>, PosHasher>> timeline(max_len);
    for (const auto& pair : paths) {
        const auto& agent_id = pair.first; const auto& path = pair.second;
        for (py::ssize_t t = 0; t < max_len; ++t) {
            if (t >= (py::ssize_t)path.size()) continue;
            timeline[t][path[t]].push_back(agent_id);
        }
    }

    // 用于记录 "这对CP" 在 "这个地点" 上次出现的时间，用于判断连续性
    std::unordered_map<std::string, int> pair_loc_last_seen;

    // Vertex Conflict 检测
    for (int t = 0; t < max_len; ++t) {
        for (const auto& [pos, agents] : timeline[t]) {
            if (agents.size() > 1) {
                for (size_t i = 0; i < agents.size(); ++i) {
                    for (size_t j = i + 1; j < agents.size(); ++j) {
                        std::string a1 = agents[i];
                        std::string a2 = agents[j];
                        
                        // >>>>> 修改点 START: 仅基于 Pos 查询是否允许汇合 <<<<<
                        bool is_intentional_loc = false;
                        
                        // 直接用 pos 去 map 里查，不需要组合时间 t
                        if (intentional_locs.count(pos)) {
                            const auto& allowed = intentional_locs.at(pos);
                            // 只有当两个智能体都在允许列表中时，才算 intentional
                            if (allowed.count(a1) && allowed.count(a2)) {
                                is_intentional_loc = true;
                            }
                        }
                        // >>>>> 修改点 END <<<<<

                        // 下面的逻辑与原函数完全一致：处理“单次连续”
                        if (is_intentional_loc) {
                            size_t pos_hash = PosHasher()(pos);
                            // Key 依然是 CP + 地点，这保证了换个地点汇合不受影响
                            std::string key = get_pair_key(a1, a2) + "_" + std::to_string(pos_hash);

                            if (pair_loc_last_seen.find(key) == pair_loc_last_seen.end()) {
                                // 第一次在这个地点见面 -> 记录
                                pair_loc_last_seen[key] = t;
                            } else {
                                int last_t = pair_loc_last_seen[key];
                                if (last_t == t - 1) {
                                    // 连续见面 -> 更新时间 (合法)
                                    pair_loc_last_seen[key] = t;
                                } else {
                                    // 曾经来过，中间断了，现在又回来了 -> 冲突 (非法重返)
                                    return py::dict("type"_a="vertex", "time"_a=t, "loc"_a=pos, "agents"_a=agents);
                                }
                            }
                        } else {
                            // 根本不在允许列表里 -> 冲突
                            return py::dict("type"_a="vertex", "time"_a=t, "loc"_a=pos, "agents"_a=agents);
                        }
                    }
                }
            }
        }
    }

    // Edge Conflict (完全一致，通常不需要豁免)
    for (auto it1 = paths.begin(); it1 != paths.end(); ++it1) {
        for (auto it2 = paths.begin(); it2 != paths.end(); ++it2) {
            if (it1->first >= it2->first) continue;
            const auto& path1 = it1->second; const auto& path2 = it2->second;
            for (py::ssize_t t = 0; t < max_len - 1; ++t) {
                if (t + 1 >= (int)path1.size() || t + 1 >= (int)path2.size()) continue;
                const Pos& u = path1[t]; const Pos& v = path1[t+1];
                const Pos& x = path2[t]; const Pos& y = path2[t+1];
                if (v == x && u == y && u != x) {
                    return py::dict("type"_a="edge", "time"_a=t+1, "loc1"_a=u, "loc2"_a=v, "agents"_a=py::make_tuple(it1->first, it2->first));
                }
            }
        }
    }
    return py::dict();
}

        // ---------------------------------------------------------
    // 1.5. 新增功能：统计所有冲突数量 (用于启发式)
    // ---------------------------------------------------------
    int count_all_conflicts(const PathMap& paths, py::dict intentional_meetings_py, py::dict /*unused*/) {
        using IntentionalMap = std::unordered_map<VertexConstraint, std::unordered_set<std::string>, VertexConstraintHasher>;
        auto intentional_meetings = intentional_meetings_py.cast<IntentionalMap>();

        int conflict_count = 0;
        py::ssize_t max_len = 0;
        if (paths.empty()) return 0;
        for (const auto& pair : paths) max_len = std::max(max_len, (py::ssize_t)pair.second.size());
        if (max_len == 0) return 0;

        // Timeline construction (与 find_conflict 相同)
        std::vector<std::unordered_map<Pos, std::vector<std::string>, PosHasher>> timeline(max_len);

        for (const auto& pair : paths) {
            const auto& agent_id = pair.first; const auto& path = pair.second;
            for (py::ssize_t t = 0; t < max_len; ++t) {
                if (t >= (py::ssize_t)path.size()) continue;
                timeline[t][path[t]].push_back(agent_id);
            }
        }

        std::unordered_map<std::string, int> pair_loc_last_seen;

        // Vertex Conflict Counting
        for (int t = 0; t < max_len; ++t) {
            for (const auto& [pos, agents] : timeline[t]) {
                if (agents.size() > 1) {
                    for (size_t i = 0; i < agents.size(); ++i) {
                        for (size_t j = i + 1; j < agents.size(); ++j) {
                            std::string a1 = agents[i];
                            std::string a2 = agents[j];
                            
                            bool is_intentional_loc = false;
                            
                            if (intentional_meetings.count({pos, t})) {
                                const auto& allowed = intentional_meetings.at({pos, t});
                                if (allowed.count(a1) && allowed.count(a2)) is_intentional_loc = true;
                            }
                            if (!is_intentional_loc && intentional_meetings.count({pos, -1})) {
                                const auto& allowed = intentional_meetings.at({pos, -1});
                                if (allowed.count(a1) && allowed.count(a2)) is_intentional_loc = true;
                            }

                            if (is_intentional_loc) {
                                size_t pos_hash = PosHasher()(pos);
                                std::string key = get_pair_key(a1, a2) + "_" + std::to_string(pos_hash);

                                if (pair_loc_last_seen.find(key) == pair_loc_last_seen.end()) {
                                    pair_loc_last_seen[key] = t;
                                } else {
                                    int last_t = pair_loc_last_seen[key];
                                    if (last_t == t - 1) {
                                        pair_loc_last_seen[key] = t;
                                    } else {
                                        // 非法二次汇合 -> 计入冲突
                                        conflict_count++;
                                        // 不更新 last_t，这样后续的非法重合时刻也会被计入冲突
                                    }
                                }
                            } else {
                                // 非白名单 -> 计入冲突
                                conflict_count++;
                            }
                        }
                    }
                }
            }
        }

        // Edge Conflict Counting
        for (auto it1 = paths.begin(); it1 != paths.end(); ++it1) {
            for (auto it2 = paths.begin(); it2 != paths.end(); ++it2) {
                if (it1->first >= it2->first) continue;
                const auto& path1 = it1->second; const auto& path2 = it2->second;
                for (py::ssize_t t = 0; t < max_len - 1; ++t) {
                    if (t + 1 >= (int)path1.size() || t + 1 >= (int)path2.size()) continue;
                    const Pos& u = path1[t]; const Pos& v = path1[t+1];
                    const Pos& x = path2[t]; const Pos& y = path2[t+1];
                    if (v == x && u == y && u != x) {
                        conflict_count++;
                    }
                }
            }
        }
        return conflict_count;
    }

    void append_path(Path& dest, const Path& src) {
        dest.reserve(dest.size() + src.size());
        dest.insert(dest.end(), src.begin(), src.end());
    }


    std::tuple<Path, Path> plan_group_paths_dynamic_t(Pos start_alpha, Pos goal_alpha, Pos start_beta, Pos pickup_beta,
                                                        Pos v_m, const py::dict& cons_alpha_py, const py::dict& cons_beta_py,
                                                        int co_work_duration, int pickup_duration, int delivery_duration) {
        // 1. 自动计时开始
        ScopedTimer timer("Dynamic_T", debug_);


        Constraints c_alpha = parse_constraints_to_cpp(cons_alpha_py);
        Constraints c_beta = parse_constraints_to_cpp(cons_beta_py);

        // ---------------------------------------------------------
        // 步骤 1: Beta 从起点到取货点
        // ---------------------------------------------------------
        Path path_b1 = space_time_a_star_core(start_beta, pickup_beta, c_beta, 0, -1, pickup_duration, false, -1);
        if (path_b1.empty()) {
            if (debug_) std::cout << "[Dynamic_T] FAILURE: Beta cannot reach pickup point from start." << std::endl;
            return {Path(), Path()};
        }
        int t_b_pickup_done = (int)path_b1.size() - 1;

        // ---------------------------------------------------------
        // 步骤 2: Beta 从取货点到汇合点 (v_m)
        // ---------------------------------------------------------
        Path path_b2 = space_time_a_star_core(pickup_beta, v_m, c_beta, t_b_pickup_done, -1, 0, false, -1);
        if (path_b2.empty()) {
            if (debug_) std::cout << "[Dynamic_T] FAILURE: Beta cannot reach meeting point (v_m) from pickup point." << std::endl;
            return {Path(), Path()};
        }

        Path path_b_to_m = path_b1; path_b_to_m.pop_back();
        append_path(path_b_to_m, path_b2);
        int t_b_at_m = (int)path_b_to_m.size() - 1;

        // [LOG] Beta路径生成成功
        if (debug_) std::cout << "[Dynamic_T] Beta path to meeting point ready. Arrives at t=" << t_b_at_m << std::endl;

        // ---------------------------------------------------------
        // 步骤 3: Alpha 从起点到汇合点 (v_m)，需避让 Beta
        // ---------------------------------------------------------
        Constraints c_alpha_dynamic = c_alpha;
        for(size_t i=0; i < path_b_to_m.size(); ++i) {
            Pos p = path_b_to_m[i];
            // 注意：Beta在最后时刻虽然停在v_m，但逻辑上Alpha也要过去，这里主要避免路径冲突
            if (p != v_m || i != path_b_to_m.size() - 1) { c_alpha_dynamic.vertex_cons.insert({p, (int)i}); }
            if (i < path_b_to_m.size() - 1) { c_alpha_dynamic.edge_cons.insert({{path_b_to_m[i+1], p}, (int)i+1}); }
        }

        Path path_a_to_m = space_time_a_star_core(start_alpha, v_m, c_alpha_dynamic, 0, -1, 0, false, -1);
        if (path_a_to_m.empty()) {
            if (debug_) std::cout << "[Dynamic_T] FAILURE: Alpha cannot reach meeting point (v_m). Possible collision with Beta or static obstacles." << std::endl;
            return {Path(), Path()};
        }
        int t_a_at_m = (int)path_a_to_m.size() - 1;

        // [LOG] Alpha路径生成成功
        if (debug_) std::cout << "[Dynamic_T] Alpha path to meeting point ready. Arrives at t=" << t_a_at_m << std::endl;

        // ---------------------------------------------------------
        // 步骤 4: 寻找同步汇合时间 (Wait Logic)
        // ---------------------------------------------------------
        int t_base = std::max(t_a_at_m, t_b_at_m);
        int t_m_found = -1;
        int t_limit = std::min(t_base + 10, default_max_time_ - co_work_duration);
        
        // [LOG] 开始搜索时间窗口
        if (debug_) std::cout << "[Dynamic_T] Searching sync time window. Base: " << t_base << ", Limit: " << t_limit << ", Co-work: " << co_work_duration << std::endl;

        for (int t = t_base; t < t_limit; ++t) {
            int t_leave = t + co_work_duration;
            bool safe_b = is_safe_to_stay_cpp(v_m, t_b_at_m, c_beta, t_leave);
            bool safe_a = is_safe_to_stay_cpp(v_m, t_a_at_m, c_alpha_dynamic, t_leave);
            
            if (safe_b && safe_a) { 
                t_m_found = t; 
                break; 
            }
        }

        if (t_m_found == -1) {
            if (debug_) {
                std::cout << "[Dynamic_T] FAILURE: Synchronization failed at v_m. Both agents arrived but cannot stay safe together." << std::endl;
                std::cout << "             Check constraints at v_m between t=" << t_base << " and " << t_limit << std::endl;
            }
            return {Path(), Path()};
        }

        // [LOG] 汇合时间找到
        if (debug_) std::cout << "[Dynamic_T] Sync time found at t=" << t_m_found << std::endl;

        // ---------------------------------------------------------
        // 步骤 5: Alpha 从汇合点 (v_m) 到 终点
        // ---------------------------------------------------------
        if (v_m != goal_alpha) {
            for(int t=t_m_found + co_work_duration + 1; t <= default_max_time_; ++t) {
                c_alpha_dynamic.vertex_cons.insert({v_m, t}); // 防止其他agent此时占用v_m (可选，视逻辑而定)
            }
        }
        
        Path path_a_from_m = space_time_a_star_core(v_m, goal_alpha, c_alpha_dynamic,
                                                    t_m_found + co_work_duration, -1, delivery_duration, false, -1);
        if (path_a_from_m.empty()) {
            if (debug_) std::cout << "[Dynamic_T] FAILURE: Alpha cannot reach final goal from meeting point." << std::endl;
            return {Path(), Path()};
        }

        // ---------------------------------------------------------
        // 拼接最终路径
        // ---------------------------------------------------------
        Path final_alpha = path_a_to_m; final_alpha.pop_back();
        for(int t=t_a_at_m; t < t_m_found + co_work_duration; ++t) final_alpha.push_back(v_m);
        append_path(final_alpha, path_a_from_m);

        Path final_beta = path_b_to_m; final_beta.pop_back();
        for(int t=t_b_at_m; t < t_m_found + co_work_duration; ++t) final_beta.push_back(v_m);

        if (debug_) std::cout << "[Dynamic_T] SUCCESS: Group paths planned successfully." << std::endl;
        return {final_alpha, final_beta};
    }

    std::tuple<Path, Path> plan_group_paths_fixed_t(Pos start_alpha, Pos goal_alpha, Pos start_beta, Pos pickup_beta,
                                                        Pos v_m, int timeWindow,
                                                        const py::dict& cons_alpha_py, const py::dict& cons_beta_py,
                                                        int co_work_duration, int pickup_duration, int delivery_duration) {
        ScopedTimer timer("Fixed_T", debug_);
        Constraints c_alpha = parse_constraints_to_cpp(cons_alpha_py);
        Constraints c_beta = parse_constraints_to_cpp(cons_beta_py);

        Path path_b_pickup = space_time_a_star_core(start_beta, pickup_beta, c_beta, 0, -1, pickup_duration, false, -1);
        if (path_b_pickup.empty()) return {Path(), Path()};
        int t_b_pickup_done = (int)path_b_pickup.size() - 1;
        Path path_b_meet_fast = space_time_a_star_core(pickup_beta, v_m, c_beta, t_b_pickup_done, -1, 0, false, -1);
        if (path_b_meet_fast.empty()) return {Path(), Path()};
        int t_b_earliest = (int)path_b_pickup.size() + (int)path_b_meet_fast.size() - 2;

        Path path_a_meet_fast = space_time_a_star_core(start_alpha, v_m, c_alpha, 0, -1, 0, false, -1);
        if (path_a_meet_fast.empty()) return {Path(), Path()};
        int t_a_earliest = (int)path_a_meet_fast.size() - 1;

        int t_base = std::max(t_a_earliest, t_b_earliest);
        int t_limit = std::min(t_base + timeWindow, default_max_time_ - co_work_duration);

        for (int t = t_base; t < t_limit; ++t) {
            int t_leave = t + co_work_duration;
            if (!is_safe_to_stay_cpp(v_m, t - 1, c_beta, t_leave)) continue;
            if (!is_safe_to_stay_cpp(v_m, t - 1, c_alpha, t_leave)) continue;

            Path path_b_jit;
            if (t == t_b_earliest) {
                path_b_jit = path_b_meet_fast;
            } else {
                path_b_jit = space_time_a_star_core(pickup_beta, v_m, c_beta, t_b_pickup_done, -1, 0, false, t);
            }
            if (path_b_jit.empty()) continue;

            Path path_b_total = path_b_pickup;
            path_b_total.pop_back();
            append_path(path_b_total, path_b_jit);

            Constraints c_alpha_jit = c_alpha;
            for(size_t i=0; i < path_b_total.size(); ++i) {
                Pos p = path_b_total[i];
                if (p != v_m || i != path_b_total.size() - 1) c_alpha_jit.vertex_cons.insert({p, (int)i});
                if (i < path_b_total.size() - 1) c_alpha_jit.edge_cons.insert({{path_b_total[i+1], p}, (int)i+1});
            }

            Path path_a_jit = space_time_a_star_core(start_alpha, v_m, c_alpha_jit, 0, -1, 0, false, t);
            if (path_a_jit.empty()) continue;

            if (v_m != goal_alpha) {
                for(int k=t_leave + 1; k <= default_max_time_; ++k) c_alpha_jit.vertex_cons.insert({v_m, k});
            }

            Path path_a_dep = space_time_a_star_core(v_m, goal_alpha, c_alpha_jit,
                                                     t_leave, -1, delivery_duration, false, -1);
            if (co_work_duration > 0) {
                for(int i=0; i<co_work_duration; ++i) 
                path_b_total.push_back(v_m);
            }
            Path path_a_total = path_a_jit; 
            path_a_total.pop_back();
            if (co_work_duration > 0) {
                 for(int i=0; i<co_work_duration; ++i) path_a_total.push_back(v_m);
            }
            append_path(path_a_total, path_a_dep);

            return {path_a_total, path_b_total};
        }

        return {Path(), Path()};
    }


    // ---------------------------------------------------------------------
    // 2. C++ 版低层规划器 (灵活版，无无限停留检查)
    // ---------------------------------------------------------------------
    std::tuple<Path, Path> plan_group_paths_flexible(
        Pos start_alpha, Pos goal_alpha, Pos start_beta, Pos pickup_beta,
        Pos v_m, int t_m_expected,
        const py::dict& cons_alpha_py, const py::dict& cons_beta_py,
        int co_work_duration, int pickup_duration, int delivery_duration) 
    {
        Constraints c_alpha = parse_constraints_to_cpp(cons_alpha_py);
        Constraints c_beta = parse_constraints_to_cpp(cons_beta_py);

        // --- 1. Beta -> Pickup ---
        Path path_b1 = space_time_a_star_cbs(start_beta, pickup_beta, c_beta, 0, -1);
        if (path_b1.empty()) return {Path(), Path()};
        int t_b_at_s = (int)path_b1.size() - 1;

        // --- 1.5. Beta Wait at Pickup ---
        int t_b_leave_s = t_b_at_s;
        if (pickup_duration > 0) {
            if (!is_safe_to_stay_cpp(pickup_beta, t_b_at_s, c_beta, t_b_at_s + pickup_duration)) {
                return {Path(), Path()};
            }
            for (int i = 0; i < pickup_duration; ++i) path_b1.push_back(pickup_beta);
            t_b_leave_s += pickup_duration;
        }

        // --- 2. Beta -> Meeting ---
        Path path_b2 = space_time_a_star_cbs(pickup_beta, v_m, c_beta, t_b_leave_s, -1);
        if (path_b2.empty()) return {Path(), Path()};

        Path path_b_to_m = path_b1;
        if (path_b2.size() > 1) {
            path_b_to_m.insert(path_b_to_m.end(), path_b2.begin() + 1, path_b2.end());
        }
        int t_b_at_m = (int)path_b_to_m.size() - 1;

        if (t_b_at_m > t_m_expected) {
            return {Path(), Path()};
        }
        // --- 3. Alpha -> Meeting ---
        Path path_a_to_m = space_time_a_star_cbs(start_alpha, v_m, c_alpha, 0, -1);
        if (path_a_to_m.empty()) return {Path(), Path()};
        int t_a_at_m = (int)path_a_to_m.size() - 1;

        // --- 4. Time Alignment ---
        int current_t_m = t_m_expected;
        int actual_arrival = std::max(t_a_at_m, t_b_at_m);
        // if (actual_arrival > current_t_m) {
        //     current_t_m = actual_arrival;
        // }
        if (t_a_at_m > current_t_m) {
            return {Path(), Path()};
        }


        // --- 5. Fill Wait Steps ---
        int wait_a = current_t_m - t_a_at_m;
        if (wait_a > 0) {
            if (!is_safe_to_stay_cpp(v_m, t_a_at_m, c_alpha, current_t_m)) return {Path(), Path()};
            for (int i = 0; i < wait_a; ++i) path_a_to_m.push_back(v_m);
        }

        int wait_b = current_t_m - t_b_at_m;
        if (wait_b > 0) {
            if (!is_safe_to_stay_cpp(v_m, t_b_at_m, c_beta, current_t_m)) return {Path(), Path()};
            for (int i = 0; i < wait_b; ++i) path_b_to_m.push_back(v_m);
        }

        // --- 6. Co-work ---
        if (co_work_duration > 0) {
            int final_meeting_time = current_t_m + co_work_duration;
            if (!is_safe_to_stay_cpp(v_m, current_t_m, c_alpha, final_meeting_time)) return {Path(), Path()};
            if (!is_safe_to_stay_cpp(v_m, current_t_m, c_beta, final_meeting_time)) return {Path(), Path()};

            for (int i = 0; i < co_work_duration; ++i) {
                path_a_to_m.push_back(v_m);
                path_b_to_m.push_back(v_m);
            }
            current_t_m = final_meeting_time; 
        }

        // --- 7. Beta Finish ---
        // if (!is_safe_to_stay_cpp(v_m, current_t_m, c_beta, -1)) return {Path(), Path()};
        Path path_beta = path_b_to_m;

        // --- 8. Alpha -> Goal ---
        Path path_a_from_m = space_time_a_star_cbs(v_m, goal_alpha, c_alpha, current_t_m, -1);
        // if (path_a_from_m.empty()) return {Path(), Path()};

        // --- 9. Alpha Delivery Wait ---
        int t_alpha_arrive_g = current_t_m + (int)path_a_from_m.size() - 1;
        int t_alpha_leave_g = t_alpha_arrive_g;

        if (delivery_duration > 0) {
            if (!is_safe_to_stay_cpp(goal_alpha, t_alpha_arrive_g, c_alpha, t_alpha_arrive_g + delivery_duration)) {
                return {Path(), Path()};
            }
            for (int i = 0; i < delivery_duration; ++i) path_a_from_m.push_back(goal_alpha);
            t_alpha_leave_g += delivery_duration;
        }

        // --- 11. Combine Alpha ---
        Path path_alpha = path_a_to_m;
        if (path_a_from_m.size() > 1) {
            path_alpha.insert(path_alpha.end(), path_a_from_m.begin() + 1, path_a_from_m.end());
        }

        return {path_alpha, path_beta};
    }


    

    // ---------------------------------------------------------
    // 3. 基础 Space-Time A* 包装器
    // ---------------------------------------------------------
    Path space_time_a_star(Pos start, Pos goal, py::dict constraints_py, int start_time, int max_time) {
        Constraints cons = parse_constraints_to_cpp(constraints_py);
        return space_time_a_star_cbs(start, goal, cons, start_time, max_time);
    }

    bool is_safe_to_stay(Pos pos, int start_time, const py::dict& constraints_py, int max_check_time = -1) {
        Constraints c = parse_constraints_to_cpp(constraints_py);
        return is_safe_to_stay_cpp(pos, start_time, c, max_check_time);
    }

    DistMap run_dijkstra(Pos start) {
        DistMap distances;
        using PQueueEntry = std::pair<int, Pos>;
        std::priority_queue<PQueueEntry, std::vector<PQueueEntry>, std::greater<PQueueEntry>> pq;
        distances[start] = 0; pq.push({0, start});
        
        while(!pq.empty()) {
            auto [dist, pos] = pq.top(); pq.pop();
            if (dist > distances[pos]) continue;
            
            for (const auto& delta : neighbors_deltas_4dir_) {
                int r = std::get<0>(pos) + std::get<0>(delta); 
                int c = std::get<1>(pos) + std::get<1>(delta);
                Pos new_pos = {r, c};
                if (!is_valid(r, c)) continue;
                int new_dist = dist + 1;
                if (!distances.count(new_pos) || new_dist < distances[new_pos]) {
                    distances[new_pos] = new_dist; pq.push({new_dist, new_pos});
                }
            }
        }
        return distances;
    }

    std::tuple<Path, int> a_star_search(Pos start, Pos goal) {
       auto dists = run_dijkstra(start);
       if(dists.find(goal) == dists.end()) return {Path(), -1};
       
       Path path; Pos curr = goal; path.push_back(curr);
       while(curr != start) {
           for(const auto& delta : neighbors_deltas_4dir_) {
               int r = std::get<0>(curr) + std::get<0>(delta);
               int c = std::get<1>(curr) + std::get<1>(delta);
               Pos prev = {r,c};
               if(dists.count(prev) && dists[prev] == dists[curr] - 1) {
                   curr = prev; path.push_back(curr); break;
               }
           }
       }
       std::reverse(path.begin(), path.end());
       return {path, dists[goal]};
    }

private:
    bool is_safe_to_stay_cpp(Pos pos, int start_time, const Constraints& c, int max_check_time = -1) {
        if (c.vertex_cons.empty()) return true;
        for (const auto& constraint : c.vertex_cons) {
            const Pos& loc = std::get<0>(constraint);
            int t = std::get<1>(constraint);
            if (loc == pos && t > start_time) {
                if (max_check_time == -1 || t <= max_check_time) return false;
            }
        }
        return true;
    }

    Path space_time_a_star_cbs(Pos start, Pos goal, const Constraints& cons, 
                                int start_time, int max_time) {
        int effective_max_time = (max_time <= 0) ? default_max_time_ : max_time;
        auto backward_h = compute_backward_heuristic(goal);
        if (backward_h[std::get<0>(start)][std::get<1>(start)] == -1) return Path();

        if (cons.vertex_cons.count({start, start_time})) return Path();

        State initial_state = {std::get<0>(start), std::get<1>(start), start_time};
        using PQueueEntry = std::tuple<double, double, State>;
        std::priority_queue<PQueueEntry, std::vector<PQueueEntry>, std::greater<PQueueEntry>> open_set;

        double h_start = (double)backward_h[std::get<0>(start)][std::get<1>(start)];
        open_set.push({h_start + (double)start_time, (double)start_time, initial_state});

        GScoreMap g_scores;
        g_scores[initial_state] = (double)start_time;
        std::unordered_map<State, State, StateHasher> came_from;
        came_from[initial_state] = initial_state;
        const double G_SCORE_INF = std::numeric_limits<double>::max();

        while (!open_set.empty()) {
            auto [curr_f, curr_g, curr_state] = open_set.top(); open_set.pop();
            Pos curr_pos = {std::get<0>(curr_state), std::get<1>(curr_state)}; 
            int curr_time = std::get<2>(curr_state);

            if (curr_g > g_scores.get(curr_state, G_SCORE_INF)) continue;

            if (curr_pos == goal) {
                Path path; State temp = curr_state;
                while (came_from.count(temp) && came_from[temp] != temp) { 
                    path.push_back({std::get<0>(temp), std::get<1>(temp)}); 
                    temp = came_from[temp]; 
                }
                path.push_back({std::get<0>(temp), std::get<1>(temp)}); 
                std::reverse(path.begin(), path.end());
                return path;
            }

            if (curr_time >= effective_max_time) continue;

            for (const auto& delta : neighbors_deltas_5dir_) {
                int r = std::get<0>(curr_pos) + std::get<0>(delta); 
                int c = std::get<1>(curr_pos) + std::get<1>(delta);
                int t_next = curr_time + 1; 
                Pos next_pos = {r, c}; 
                State next_state = {r, c, t_next};
                
                if (!is_valid(r, c) || backward_h[r][c] == -1) continue;

                if (cons.vertex_cons.count({next_pos, t_next})) continue;
                if (cons.edge_cons.count({{curr_pos, next_pos}, t_next})) continue;
                if (cons.edge_cons.count({{next_pos, curr_pos}, t_next})) continue;

                double new_g = curr_g + 1.0;
                if (new_g < g_scores.get(next_state, G_SCORE_INF)) {
                    g_scores[next_state] = new_g; 
                    came_from[next_state] = curr_state;
                    open_set.push({new_g + (double)backward_h[r][c], new_g, next_state});
                }
            }
        }
        return Path();
    }

        // --- Core A* ---
    Path space_time_a_star_core(Pos start, Pos goal, const Constraints& cons,
                              int start_time, int max_time, int work_duration,
                              bool safe_for_indefinite_stay, int min_arrival_time) {

        int effective_max_time = (max_time <= 0) ? default_max_time_ : max_time;
        auto backward_h = compute_backward_heuristic(goal);
        if (backward_h[std::get<0>(start)][std::get<1>(start)] == -1) return Path();

        int last_constraint_time_on_goal = -1;
        for (const auto& vc : cons.vertex_cons) {
            if (std::get<0>(vc) == goal) { last_constraint_time_on_goal = std::max(last_constraint_time_on_goal, std::get<1>(vc)); }
        }

        if (cons.vertex_cons.count({start, start_time})) return Path();

        State initial_state = {std::get<0>(start), std::get<1>(start), start_time};
        using PQueueEntry = std::tuple<double, double, State>;
        std::priority_queue<PQueueEntry, std::vector<PQueueEntry>, std::greater<PQueueEntry>> open_set;

        double h_start = (double)backward_h[std::get<0>(start)][std::get<1>(start)];
        open_set.push({h_start + (double)start_time, (double)start_time, initial_state});

        std::unordered_map<State, State, StateHasher> came_from;
        GScoreMap g_scores;
        came_from[initial_state] = initial_state; g_scores[initial_state] = (double)start_time;
        const double G_SCORE_INF = std::numeric_limits<double>::max();

        int expansions = 0;
        while (!open_set.empty()) {
            auto [curr_f, curr_g, curr_state] = open_set.top(); open_set.pop();
            expansions++;
            // if (expansions > 10000000) {
            //     if (debug_)
            //     {
            //         std::cout << "[A* CORE] Terminating search due to excessive expansions (>5,000,000)." << std::endl;
            //     }
            //     return Path();
                
            // }

            Pos curr_pos = {std::get<0>(curr_state), std::get<1>(curr_state)}; int curr_time = std::get<2>(curr_state);
            if (curr_g > g_scores.get(curr_state, G_SCORE_INF)) continue;

            if (curr_pos == goal) {
                if (min_arrival_time != -1 && curr_time < min_arrival_time) {
                     // Wait
                } else {
                    bool safe = true;
                    if (work_duration > 0) {
                        for(int t = 1; t <= work_duration; ++t) {
                           if (cons.vertex_cons.count({goal, curr_time + t})) { safe = false; break; }
                        }
                    }
                    // if (safe && (safe_for_indefinite_stay || work_duration == -1)) {
                    //     int finish_time = curr_time + (work_duration > 0 ? work_duration : 0);
                    //     if (finish_time <= last_constraint_time_on_goal) safe = false;
                    // }
                    if (safe) {
                        Path path; State temp = curr_state;
                        while (came_from.count(temp) && came_from[temp] != temp) { path.push_back({std::get<0>(temp), std::get<1>(temp)}); temp = came_from[temp]; }
                        path.push_back({std::get<0>(temp), std::get<1>(temp)}); std::reverse(path.begin(), path.end());
                        if (work_duration > 0) { for(int i=0; i<work_duration; ++i) path.push_back(goal); }
                        return path;
                    }
                }
            }

            if (curr_time >= effective_max_time) continue;

            for (const auto& delta : neighbors_deltas_5dir_) {
                int r = std::get<0>(curr_pos) + std::get<0>(delta); int c = std::get<1>(curr_pos) + std::get<1>(delta);
                int t_next = curr_time + 1; Pos next_pos = {r, c}; State next_state = {r, c, t_next};
                if (!is_valid(r, c) || backward_h[r][c] == -1) continue;

                if (min_arrival_time != -1 && (double)t_next + (double)backward_h[r][c] > (double)min_arrival_time) continue;

                if (cons.vertex_cons.count({next_pos, t_next})) continue;
                if (cons.edge_cons.count({{curr_pos, next_pos}, t_next})) continue;
                if (cons.edge_cons.count({{next_pos, curr_pos}, t_next})) continue;

                double move_cost = 1.0;
                if (cons.has_heatmap) move_cost += cons.heatmap[r][c];

                double new_g = curr_g + move_cost;
                if (new_g < g_scores.get(next_state, G_SCORE_INF)) {
                    g_scores[next_state] = new_g; came_from[next_state] = curr_state;
                    open_set.push({new_g + (double)backward_h[r][c], new_g, next_state});
                }
            }
        }
        return Path();
    }


    py::ssize_t height_; py::ssize_t width_; std::vector<std::vector<int>> grid_; int default_max_time_; bool debug_;
    const std::vector<Pos> neighbors_deltas_5dir_ = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {0, 0}};
    const std::vector<Pos> neighbors_deltas_4dir_ = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    bool is_valid(int r, int c) { return (r >= 0 && r < height_ && c >= 0 && c < width_ && grid_[r][c] != 1); }

    std::vector<std::vector<int>> compute_backward_heuristic(Pos target) {
        std::vector<std::vector<int>> h_map(height_, std::vector<int>(width_, -1));
        std::queue<Pos> q;
        int tr = std::get<0>(target); int tc = std::get<1>(target);
        if (grid_[tr][tc] != 0) return h_map;
        h_map[tr][tc] = 0; q.push(target);
        while(!q.empty()){
            Pos curr = q.front(); q.pop();
            int r = std::get<0>(curr); int c = std::get<1>(curr); int dist = h_map[r][c];
            for(auto& d : neighbors_deltas_4dir_) {
                int nr = r + std::get<0>(d); int nc = c + std::get<1>(d);
                if(is_valid(nr, nc) && h_map[nr][nc] == -1) { h_map[nr][nc] = dist + 1; q.push({nr, nc}); }
            }
        }
        return h_map;
    }
};

PYBIND11_MODULE(planner_lib, m) {
    py::class_<planner_lib>(m, "planner_lib")
        .def(py::init<py::array_t<int>, int, bool>(), 
             py::arg("grid"), py::arg("default_max_time")=800, py::arg("debug")=false)
        .def("space_time_a_star", &planner_lib::space_time_a_star, 
             py::arg("start"), py::arg("goal"), py::arg("constraints"), py::arg("start_time")=0, py::arg("max_time")=-1)
        .def("find_conflict", &planner_lib::find_conflict)
        .def("is_safe_to_stay", &planner_lib::is_safe_to_stay)
        .def("a_star_search", &planner_lib::a_star_search)
        .def("run_dijkstra", &planner_lib::run_dijkstra)
        .def("plan_group_paths_flexible", &planner_lib::plan_group_paths_flexible,
             py::arg("start_alpha"), py::arg("goal_alpha"), 
             py::arg("start_beta"), py::arg("pickup_beta"),
             py::arg("v_m"), py::arg("t_m_expected"),
             py::arg("cons_alpha"), py::arg("cons_beta"),
             py::arg("co_work_duration"), py::arg("pickup_duration"), py::arg("delivery_duration"))
        .def("plan_group_paths_dynamic_t", &planner_lib::plan_group_paths_dynamic_t,
             py::arg("start_alpha"), py::arg("goal_alpha"), py::arg("start_beta"), py::arg("pickup_beta"),
             py::arg("v_m"), py::arg("cons_alpha"), py::arg("cons_beta"),
             py::arg("co_work_duration"), py::arg("pickup_duration"), py::arg("delivery_duration"))
        .def("plan_group_paths_fixed_t", &planner_lib::plan_group_paths_fixed_t,
             py::arg("start_alpha"), py::arg("goal_alpha"), py::arg("start_beta"), py::arg("pickup_beta"),
             py::arg("v_m"), py::arg("t_m"),
             py::arg("cons_alpha"), py::arg("cons_beta"),
             py::arg("co_work_duration"), py::arg("pickup_duration"), py::arg("delivery_duration"))
        .def("count_all_conflicts", &planner_lib::count_all_conflicts)
        // 这里的 & 后面必须加上 planner_lib::
        .def("find_conflict_spatial", &planner_lib::find_conflict_spatial, "New function with location-only constraints");
}
