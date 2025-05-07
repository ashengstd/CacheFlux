import numpy as np
from pulp import (
    PULP_CBC_CMD,
    LpAffineExpression,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

from utils.config import LP_LOG_PATH, PULP_LOG_ENABLE, THREADS


def objective_function(
    B: np.ndarray, A_ij: np.ndarray, cost: np.ndarray, users: int, caches: int
) -> LpAffineExpression:
    return lpSum(
        A_ij[i][j] * B[i][j] * cost[j] for i in range(users) for j in range(caches)
    )


def objective_function_new(
    B: np.ndarray,
    A_ij: np.ndarray,
    node_cost: np.ndarray,
    users: int,
    caches: int,
    mappings: np.ndarray,
) -> LpAffineExpression:
    pass
    # 计算节点的流量分配情况，并存储为pulp的变量
    node_flows = np.sum(np.array(B) @ mappings.T, axis=0)
    exclude_index = np.argsort(node_flows)[-3:]
    node_flows[exclude_index] = 0
    return lpSum(node_flows[i] * node_cost[i] for i in range(node_cost.shape[0]))


def constraint1(
    B: np.ndarray, I_ij: np.ndarray, R_i: np.ndarray, users: int, caches: int
) -> list[LpAffineExpression]:
    constraints = []
    for i in range(users):
        constraint = lpSum(I_ij[i][j] * B[i][j] for j in range(caches)) >= R_i[i]
        constraints.append(constraint)
    return constraints


def constraint2(
    B: np.ndarray, I_ij: np.ndarray, max_bandwidth: np.ndarray, users: int, caches: int
) -> list[LpAffineExpression]:
    constraints = []
    for j in range(caches):
        constraint = (
            lpSum(I_ij[i][j] * B[i][j] for i in range(users)) <= max_bandwidth[j]
        )
        constraints.append(constraint)
    return constraints


def max_constraint(
    B: np.ndarray,
    I_ij: np.ndarray,
    max_values: np.ndarray,
    mapping: np.ndarray,
    users: int,
    caches: int,
) -> list[LpAffineExpression]:
    constraints = []
    nodes = max_values.shape[0]
    node_indices = [np.where(mapping[node_idx] == 1)[0] for node_idx in range(nodes)]
    for node_idx in range(nodes):
        for i in range(users):
            constraint = (
                lpSum(I_ij[i][j] * B[i][j] for j in node_indices[node_idx])
                <= max_values[node_idx]
            )
            constraints.append(constraint)
    return constraints


def solve_optimization(
    I_ij: np.ndarray,
    R_i: np.ndarray,
    A_ij: np.ndarray,
    max_values: np.ndarray,
    max_bandwidth: np.ndarray,
    mapping: np.ndarray,
    cost: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    求解优化问题并返回结果, Shape为 USERS*CACHES
    """
    prob = LpProblem("Minimize_B", LpMinimize)
    users, caches = I_ij.shape[0], I_ij.shape[1]
    B = LpVariable.matrix("B", (range(users), range(caches)), lowBound=0)

    # 设置目标函数
    prob += objective_function_new(B, A_ij, cost, users, caches, mapping)

    # 添加约束条件
    for constraint in constraint1(B, I_ij, R_i, users, caches):
        prob += constraint
    for constraint in constraint2(B, I_ij, max_bandwidth, users, caches):
        prob += constraint
    for constraint in max_constraint(B, I_ij, max_values, mapping, users, caches):
        prob += constraint

    # 求解
    prob.solve(PULP_CBC_CMD(msg=0, threads=THREADS))

    # 记录求解状态
    if PULP_LOG_ENABLE:
        with open(LP_LOG_PATH, "a") as log_file:
            status = "Optimal" if prob.status == 1 else "Infeasible"
            log_file.write(f"Status: {status}\n")

    return np.array(
        [
            [value(B[i][j]) if value(B[i][j]) is not None else 0 for j in range(caches)]
            for i in range(users)
        ],
        dtype=np.int32,
    ), prob.status
