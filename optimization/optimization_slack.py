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

from optimization.SharedMemory import SharedMemManager

PUNISHMENT = 1e3


def objective_function(
    B: np.ndarray,
    A_ij: np.ndarray,
    cost: np.ndarray,
    users: int,
    caches: int,
    slack_vars: list[LpAffineExpression],
) -> LpAffineExpression:
    return (
        lpSum(
            A_ij[i][j] * B[i][j] * cost[j] for i in range(users) for j in range(caches)
        )
        + lpSum(slack_vars) * PUNISHMENT
    )


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


def min_constraint_with_slack(
    B: np.ndarray,
    I_ij: np.ndarray,
    min_values: np.ndarray,
    mapping: np.ndarray,
    users: int,
    caches: int,
) -> tuple[list[LpAffineExpression], list[LpAffineExpression]]:
    constraints = []
    slack_vars = []  # 存储松弛变量

    nodes = min_values.shape[0]
    node_indices = [np.where(mapping[node_idx] == 1)[0] for node_idx in range(nodes)]

    for node_idx in range(nodes):
        for i in range(users):
            # 创建松弛变量
            slack_var = LpVariable(
                f"slack_{node_idx}_{i}", lowBound=0
            )  # 松弛变量不小于0
            slack_vars.append(slack_var)

            # 添加约束，最小值的松弛约束
            constraint = (
                lpSum(I_ij[i][j] * B[i][j] for j in node_indices[node_idx]) - slack_var
                >= min_values[node_idx]
            )
            constraints.append(constraint)

    return constraints, slack_vars


def max_constraint_with_slack(
    B: np.ndarray,
    I_ij: np.ndarray,
    max_values: np.ndarray,
    mapping: np.ndarray,
    users: int,
) -> tuple[list[LpAffineExpression], list[LpAffineExpression]]:
    constraints = []
    slack_vars = []  # 存储松弛变量

    nodes = max_values.shape[0]
    node_indices = [np.where(mapping[node_idx] == 1)[0] for node_idx in range(nodes)]

    for node_idx in range(nodes):
        for i in range(users):
            # 创建松弛变量
            slack_var = LpVariable(
                f"slack_{node_idx}_{i}", lowBound=0
            )  # 松弛变量不小于0
            slack_vars.append(slack_var)

            # 添加约束，松弛后的约束条件
            constraint = (
                lpSum(I_ij[i][j] * B[i][j] for j in node_indices[node_idx]) + slack_var
                <= max_values[node_idx]
            )
            constraints.append(constraint)

    return constraints, slack_vars


def lp(idx) -> tuple[np.ndarray, int]:
    """
    求解优化问题并返回结果, Shape为 USERS*CACHES
    """

    I_ij = SharedMemManager.get_by_name("I_ij")
    A_ij = SharedMemManager.get_by_name("A_ij")
    R_i = SharedMemManager.get_by_name("R")
    max_values = SharedMemManager.get_by_name("max_values")
    max_bandwidths = SharedMemManager.get_by_name("max_bandwidths")
    mappings = SharedMemManager.get_by_name("mappings")
    costs = SharedMemManager.get_by_name("costs")

    assert isinstance(I_ij, np.ndarray)
    assert isinstance(A_ij, np.ndarray)
    assert isinstance(R_i, np.ndarray)
    assert isinstance(max_values, np.ndarray)
    assert isinstance(max_bandwidths, np.ndarray)
    assert isinstance(mappings, np.ndarray)
    assert isinstance(costs, np.ndarray)

    I_ij = I_ij[:, idx, :]
    A_ij = A_ij[:, idx, :]

    prob = LpProblem("Minimize_B", LpMinimize)
    users, caches = I_ij.shape[0], I_ij.shape[1]
    B = LpVariable.matrix("B", (range(users), range(caches)), lowBound=0)

    # 添加约束条件
    for constraint in constraint1(B, I_ij, R_i, users, caches):
        prob += constraint
    for constraint in constraint2(B, I_ij, max_bandwidths, users, caches):
        prob += constraint
    # constraints_min, slack_vars_min = min_constraint_with_slack(B, I_ij, min_values, mapping, users, caches)
    # for constraint in constraints_min:
    #     prob += constraint
    constraints_max, slack_vars_max = max_constraint_with_slack(
        B, I_ij, max_values, mappings, users
    )
    for constraint in constraints_max:
        prob += constraint
    # slack_vars = [slack_vars_min, slack_vars_max]
    slack_vars = slack_vars_max
    # 设置目标函数
    prob += objective_function(B, A_ij, costs, users, caches, slack_vars)

    # 求解
    prob.solve(PULP_CBC_CMD(msg=0))

    return np.array(
        [
            [value(B[i][j]) if value(B[i][j]) is not None else 0 for j in range(caches)]
            for i in range(users)
        ],
        dtype=np.int32,
    ), prob.status
