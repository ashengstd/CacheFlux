from multiprocessing import Pool
from typing import Tuple

import numpy as np
from optimization.optimization import solve_optimization
from rich.progress import Progress


def process_solution(
    args: Tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
):
    i, I_ij, R, A_ij, max_values, max_bandwidth, mapping, cost = args
    I_ij_i: np.ndarray = I_ij[:, i, :]
    A_ij_i: np.ndarray = A_ij[:, i, :]

    # 求解优化问题
    B: np.ndarray
    B, status = solve_optimization(
        I_ij_i, R, A_ij_i, max_values, max_bandwidth, mapping, cost
    )
    return i, B, status


def compute_solutions(
    R: np.ndarray,
    I_ij: np.ndarray,
    A_ij: np.ndarray,
    max_values: np.ndarray,
    max_bandwidth: np.ndarray,
    mapping: np.ndarray,
    cost: np.ndarray,
    progress: Progress | None,
) -> np.ndarray:
    """计算给定时间点的优化方案"""
    # 方案数量在变量shape中间，所以shape[1]是方案数量
    # 下面创建一个shape为 (方案数量, 用户数量, 缓存数量) 的数组
    nums = I_ij.shape[1]
    solutions = []
    if progress:
        task = progress.add_task("计算所有方案...", total=nums)

    def update_progress(result):
        _, B, status = result
        if status == 1:
            solutions.append(B)
        if progress is not None:
            progress.update(task, advance=1)

    with Pool() as pool:
        args = [
            (i, I_ij, R, A_ij, max_values, max_bandwidth, mapping, cost)
            for i in range(nums)
        ]
        for arg in args:
            pool.apply_async(process_solution, args=(arg,), callback=update_progress)
        pool.close()
        pool.join()

    if progress is not None:
        progress.remove_task(task)
    return np.array(solutions)


def get_best_solution(
    solutions: np.ndarray,
    mappings: np.ndarray,
    cost: np.ndarray,
    progress: Progress | None,
) -> tuple[int, np.ndarray]:
    """获取最优方案"""
    min_fitness = float("inf")
    # 方案数量在变量shape中间，所以shape[1]是方案数量
    # 即创建一个shape为 (用户数量, 缓存数量) 的数组
    users = solutions.shape[1]
    caches = solutions.shape[2]
    best_solution: np.ndarray = np.zeros((users, caches))
    best_index = 0
    if progress is not None:
        solution_task = progress.add_task("寻找最优方案...", total=solutions.shape[0])

    for i in range(solutions.shape[0]):
        # 循环所有分配方案
        B = solutions[i, :, :]
        if np.all(B == 0):
            continue

        # 计算节点的流量
        node_flows = np.sum(B @ mappings.T, axis=0)

        # 扔掉流量最大的三个节点
        exclude_index = np.argsort(node_flows)[-3:]
        node_flows[exclude_index] = 0
        fitness = np.sum(node_flows @ cost)  # 适应度值计算

        # 将适应度值为0的方案删除
        if fitness == 0:
            continue

        # 更新最小适应度值和最优方案
        if fitness < min_fitness:
            min_fitness = fitness
            best_index = i
            best_solution = B  # 保存最优方案
        if progress is not None:
            progress.update(solution_task, advance=1)
    if progress is not None:
        progress.remove_task(solution_task)
    return best_index, best_solution
