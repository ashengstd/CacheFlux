import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models import UserReq
from models.plNetwork import MemoryConfig, plMemoryDNN
from optimization.solutions import compute_solutions, get_best_solution
from utils.config import (
    CACHES,
    INFO_CSV_PATH,
    INFO_NPY_PATH,
    LOG_PATH,
    LP_LOG_PATH,
    MODEL_SAVE_PATH,
    PL_PARAMS,
    REQ_CSV_CSV_PATH,
    TEST_SOLUTION_PATH,
)
from utils.logger import logger


def prepareDirectories():
    """Create necessary directories and clean up log files."""
    for path in [LOG_PATH, TEST_SOLUTION_PATH]:
        Path(path).mkdir(parents=True, exist_ok=True)

    if LP_LOG_PATH.exists():
        LP_LOG_PATH.unlink()


def loadPreprocessedData():
    """Load preprocessed data from specified paths."""
    max_values = np.load(INFO_NPY_PATH.joinpath("upper.npy"))
    max_bandwidth = np.load(INFO_NPY_PATH.joinpath("max_bandwidths.npy"))
    mapping = np.load(INFO_NPY_PATH.joinpath("node_cache_matrix.npy"))
    cost = np.load(INFO_NPY_PATH.joinpath("cost.npy"))
    node_costs = np.load(INFO_NPY_PATH.joinpath("node_costs.npy"))

    with open(INFO_NPY_PATH.joinpath("pre_data.json"), encoding="UTF-8") as f:
        cache2id = json.load(f)["cache2id"]

    return max_values, max_bandwidth, mapping, cost, node_costs, cache2id


def loadDataframeFiles():
    """Load CSV files into pandas DataFrames."""
    file_names = [
        "coverage_cache_group_info.csv",
        "cache_group_info.csv",
        "node_info.csv",
        "coverage_info.csv",
        "quality_level_info.csv",
    ]
    return {
        "coverage_cache_group_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[0])),
        "cache_group_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[1])),
        "node_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[2])),
        "coverage_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[3])),
        "quality_level_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[4])),
    }


def processUser(args):
    i, user, dataframes, cache2id, CACHES = args
    req, connectivity = user.get_connectivity(
        dataframes["coverage_cache_group_info"],
        dataframes["cache_group_info"],
        dataframes["node_info"],
        dataframes["coverage_info"],
        dataframes["quality_level_info"],
        cache2id,
        CACHES,
    )
    return i, req, connectivity


def drooPulpPipeline(
    R,
    droo_input,
    N,
    model,
    max_values,
    max_bandwidth,
    mappings,
    cost,
    node_cost,
):
    droo_output = np.array(model.decode(droo_input, N))
    droo_connect = droo_output - droo_input[:, np.newaxis, :] + 1
    solutions = compute_solutions(
        R,
        droo_output,
        droo_connect,
        max_values,
        max_bandwidth,
        mappings,
        cost,
    )

    if solutions.shape[0] == 0:
        return 0, None, float("inf")

    _, min_cost, best_solution = get_best_solution(solutions, mappings, node_cost)
    return solutions.shape[0], best_solution, min_cost


def lp_pipeline(
    R,
    droo_input,
    max_values,
    max_bandwidth,
    mappings,
    cost,
    node_cost,
):
    solutions = compute_solutions(
        R,
        droo_input,
        droo_input,
        max_values,
        max_bandwidth,
        mappings,
        cost,
    )

    if solutions.shape[0] == 0:
        return 0, None, float("inf")

    _, min_cost, best_solution = get_best_solution(solutions, mappings, node_cost)
    return solutions.shape[0], best_solution, min_cost


def getRequestsAndConnectivity(user_list, cache2id, dataframes):
    users = len(user_list)
    connectivity_matrix = np.zeros((users, CACHES), dtype=bool)
    valid_users = []
    UserReqs = np.zeros(users, dtype=int)
    logger.info("Start processing user requests and connectivity matrix...")

    # 使用多进程处理用户连通性
    def collect_result(result):
        i, req, connectivity = result
        UserReqs[i] = req
        connectivity_matrix[i, :] = connectivity
        valid_users.append(i)

    for i, user in enumerate(user_list):
        args = (i, user, dataframes, cache2id, CACHES)
        result = processUser(args)
        collect_result(result)
    return UserReqs[valid_users], connectivity_matrix[valid_users, :]


def inferWithRetry(
    R,
    connectivity_matrix,
    N,
    model,
    max_values,
    max_bandwidth,
    mappings,
    cost,
    node_cost,
):
    solution_nums = 0
    best_solution = None
    retry = math.floor(math.log2(CACHES / N)) + 1
    current = 0
    while current <= retry:
        current += 1
        logger.info(f"retry for the {current}th time")
        solution_nums, best_solution, min_cost = drooPulpPipeline(
            R,
            connectivity_matrix,
            2 ** (current - 1) * N if 2 ** (current - 1) * N < CACHES else CACHES,
            model,
            max_values,
            max_bandwidth,
            mappings,
            cost,
            node_cost,
        )
        if solution_nums > 4:
            break
        logger.debug(
            f"Retrying for the {current}th time, solution numbers: {solution_nums}, current solution numbers: {2 ** (current - 1) * N}"
        )
    else:
        if solution_nums == 0:
            logger.error("No solution found after retrying")
        return
    logger.info(
        f"Found {solution_nums} solutions, best solution found, saved, min cost: {min_cost:.2f}, solution numbers: {solution_nums}"
    )
    return best_solution, min_cost


def infer_csv_pipeline(
    model,
    test_csv: Path,
    timepoint: int = 167,
    N=16,
    max_values=None,
    max_bandwidth=None,
    mappings=None,
    cost=None,
    node_cost=None,
    cache2id=None,
    dataframes=None,
):
    # 加载用户需求
    df = pd.read_csv(test_csv)
    df.rename(columns={"6月用户带宽数据": "带宽数据"}, inplace=True)
    df = df[df["时间点"] == timepoint]
    # 初始化进度条和连接矩阵
    user_list = []
    for row in df.itertuples(index=False):
        user_list.append(
            UserReq(
                province=str(row.省份),
                operator=str(row.运营商),
                coverage_name=str(row.覆盖名),
                ip_type=int(row.IP类型),
                reqs=int(row.带宽数据),
            )
        )
    UserReqs, connectivity_matrix = getRequestsAndConnectivity(
        user_list, cache2id, dataframes
    )
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
        max_values=max_values,
        max_bandwidth=max_bandwidth,
        mappings=mappings,
        cost=cost,
        node_cost=node_cost,
    )
    if best_solution is None:
        logger.error("No solution found")
    else:
        logger.info(
            f"DRLOP time cost：{time.time() - begin:.2f}second(s), Minimal Cost：{min_cost:.2f}"
        )
    begin = time.time()
    _, best_solution, min_cost = lp_pipeline(
        R=UserReqs,
        droo_input=connectivity_matrix,
        max_values=max_values,
        max_bandwidth=max_bandwidth,
        mappings=mappings,
        cost=cost,
        node_cost=node_cost,
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

    # 文件夹准备和数据加载
    prepareDirectories()
    max_values, max_bandwidth, mappings, cost, node_cost, cache2id = (
        loadPreprocessedData()
    )
    dataframes = loadDataframeFiles()

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
                max_values=max_values,
                max_bandwidth=max_bandwidth,
                mappings=mappings,
                cost=cost,
                node_cost=node_cost,
                cache2id=cache2id,
                dataframes=dataframes,
            )


if __name__ == "__main__":
    random_test()
