from multiprocessing import Pool

import numpy as np

from optimization.optimization_slack import solve_optimization
from optimization.SharedMemory import SharedMemManager
from utils.logger import logger


def process_solution_shm(idx):
    logger.debug(f"Processing solution {idx}...")

    B, status = solve_optimization(idx=idx)

    return idx, B, status


def compute_solutions_shm(
    R: np.ndarray,
    I_ij: np.ndarray,
    A_ij: np.ndarray,
):
    """计算并更新共享内存中的解决方案"""
    solutions = []

    # 创建共享内存
    data_dict = {
        "I_ij": I_ij,
        "A_ij": A_ij,
        "R": R,
    }

    logger.info("Creating shared memory blocks for solutions...")
    shm_blocks = {}  # 保存共享内存块
    for name, data in data_dict.items():
        shm_name = SharedMemManager.create_block(name, data)
        shm_blocks[name] = shm_name

    # 检查所有共享内存块的创建情况
    for name, shm_name in shm_blocks.items():
        logger.info(f"Shared memory block {name} created")

    nums = I_ij.shape[1]
    logger.info(f"Calculating all solutions... Number of solutions: {nums}")

    def update_solutions(result):
        _, B, status = result
        if status == 1:
            solutions.append(B)

    def handle_error(e):
        logger.error(f"Error in child process: {e}")

    # 进行并行计算
    with Pool() as pool:
        args = [i for i in range(nums)]
        for arg in args:
            pool.apply_async(
                process_solution_shm,
                args=(arg,),
                callback=update_solutions,
                error_callback=handle_error,
            )
        pool.close()
        pool.join()

    # 清理共享内存
    for name in shm_blocks.keys():
        SharedMemManager.delete_block(name)

    return np.array(solutions)


def get_best_solution(
    solutions: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """获取最优方案"""
    min_fitness = float("inf")
    # 方案数量在变量shape中间，所以shape[1]是方案数量
    # 即创建一个shape为 (用户数量, 缓存数量) 的数组
    users = solutions.shape[1]
    caches = solutions.shape[2]
    best_solution: np.ndarray = np.zeros((users, caches))
    best_index = 0
    logger.info(
        f"Search for the best solution... Number of solutions: {solutions.shape[0]}"
    )

    mappings, _ = SharedMemManager.get("mappings")
    node_costs, _ = SharedMemManager.get("node_costs")
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
        fitness = np.sum(node_flows @ node_costs)  # 适应度值计算

        # 将适应度值为0的方案删除
        if fitness == 0:
            continue

        # 更新最小适应度值和最优方案
        if fitness < min_fitness:
            min_fitness = fitness
            best_index = i
            best_solution = B  # 保存最优方案

    return best_index, min_fitness, best_solution
