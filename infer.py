import math
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from models import UserReq
from models.plNetwork import MemoryConfig, plMemoryDNN
from optimization.solutions import compute_solutions, get_best_solution
from utils.config import (
    CACHES,
    MODEL_SAVE_PATH,
    PL_PARAMS,
    REQ_CSV_CSV_PATH,
)
from utils.func import load_preprocessed_npy_data, loadInfoData
from utils.logger import logger


def drooPulpPipeline(
    R: np.ndarray,
    droo_input: np.ndarray,
    N: int,
    model: plMemoryDNN,
) -> Tuple[int, np.ndarray, float]:
    I_ij = np.array(model.decode(droo_input, N), dtype=np.int8)
    A_ij = I_ij - droo_input[:, np.newaxis, :] + 1

    solutions = compute_solutions(
        R,
        I_ij,
        A_ij,
    )

    if solutions.shape[0] == 0:
        return 0, np.array([]), float("inf")

    _, min_cost, best_solution = get_best_solution(solutions)
    return solutions.shape[0], best_solution, min_cost


def lp_pipeline(
    R: np.ndarray,
    I_ij: np.ndarray,
):
    """Main pipeline for the LP method."""
    I_ij = np.array(I_ij[:, np.newaxis, :], dtype=np.int8)

    solutions = compute_solutions(
        R,
        I_ij,
        I_ij,
    )

    if solutions.shape[0] == 0:
        return 0, None, float("inf")

    _, min_cost, best_solution = get_best_solution(solutions)
    return solutions.shape[0], best_solution, min_cost


def inferWithRetry(
    R: np.ndarray,
    connectivity_matrix: np.ndarray,
    N: int,
    model: plMemoryDNN,
) -> Tuple[np.ndarray, float]:
    solution_nums = 0
    best_solution = None
    retry = math.floor(math.log2(CACHES / N)) + 1
    current = 0
    while current <= retry:
        n = min(int(2 ** (current - 1) * N), CACHES)
        current += 1
        logger.info(f"retry for the {current}th time")
        solution_nums, best_solution, min_cost = drooPulpPipeline(
            R,
            connectivity_matrix,
            N=n,
            model=model,
        )
        if solution_nums > 4:
            break
        logger.debug(
            f"Retrying for the {current}th time, solution numbers: {solution_nums}, current solution numbers: {2 ** (current - 1) * N}"
        )

    else:
        if solution_nums == 0:
            logger.error("No solution found after retrying")
        return np.array([]), float("inf")
    logger.info(
        f"Found {solution_nums} solutions, best solution found, saved, min cost: {min_cost:.2f}, solution numbers: {solution_nums}"
    )
    return best_solution, min_cost


def infer_csv_pipeline(
    model,
    test_csv: Path,
    timepoint: int = 167,
    N=16,
) -> None:
    # 加载用户需求
    df = pd.read_csv(test_csv)
    df.rename(columns={"6月用户带宽数据": "带宽数据"}, inplace=True)
    df = df[df["时间点"] == timepoint]
    # 初始化进度条和连接矩阵
    user_list = []
    for row in df.itertuples(index=False):
        user_list.append(UserReq.from_row(row=row))
    # 过滤掉带宽数据为0的用户
    user_list = [
        user for user in user_list if user.reqs > 0 and user.get_connectivity().size > 0
    ]
    UserReqs, connectivity_matrix = UserReq.getRequestsAndConnectivity(user_list)
    if UserReqs.shape[0] == 0:
        logger.error("No valid user request found")
        return
    begin = time.time()
    best_solution: np.ndarray | None
    min_cost: float
    best_solution, min_cost = inferWithRetry(
        R=UserReqs,
        connectivity_matrix=connectivity_matrix,
        N=N,
        model=model,
    )
    if best_solution.size == 0:
        logger.error("No solution found")
    else:
        logger.info(
            f"DRLOP time cost：{time.time() - begin:.2f}second(s), Minimal Cost：{min_cost:.2f}"
        )
    begin = time.time()
    _, best_solution, min_cost = lp_pipeline(
        R=UserReqs,
        I_ij=connectivity_matrix,
    )
    if best_solution is None:
        logger.error("No solution found")
    else:
        logger.info(
            f"LP time cost：{time.time() - begin:.2f}second(s), Minimal Cost：{min_cost:.2f}"
        )


def random_test():
    # 初始化模型参数
    memory_config = MemoryConfig(**PL_PARAMS)
    MemoryDNN_Net = plMemoryDNN(memory_config)
    MemoryDNN_Net.load_model(MODEL_SAVE_PATH.joinpath("best"))

    # load preprocessed data
    load_preprocessed_npy_data()
    loadInfoData()

    # 获取所有可用文件
    all_files_5 = list(REQ_CSV_CSV_PATH.joinpath("5").iterdir())
    all_files_6 = list(REQ_CSV_CSV_PATH.joinpath("6").iterdir())
    all_files = all_files_5 + all_files_6

    # 生成与每个文件对应的时间点
    base_time = 228
    bias = 12
    file_timepoints = [
        (
            file,
            (
                random.randint(base_time + bias * i, base_time + bias * (i + 1))
                for i in range(8)
                for _ in range(4)
            ),
        )
        for file in all_files
    ]

    for random_date_csv, timepoints in file_timepoints:
        for timepoint in list(timepoints):
            infer_csv_pipeline(
                model=MemoryDNN_Net,
                test_csv=random_date_csv,
                timepoint=timepoint,
                N=16,
            )


if __name__ == "__main__":
    random_test()
