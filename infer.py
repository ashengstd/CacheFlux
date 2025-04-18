import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from optimization.solutions import compute_solutions, get_best_solution
from rich.console import Console
from rich.progress import Progress
from utils.constants import (
    CACHES,
    DATA_PATH,
    LOG_PATH,
    MEMORY_DNN_LOG_PATH,
    MODEL_SAVE_PATH,
    PARAMS,
    PRE_DATA_PATH,
    SIMPLEX_LOG_PATH,
    TEST_SOLUTION_PATH,
)

from models import User_Req
from models.Network import MemoryDNN


def prepareDirectories():
    """准备所需文件夹并清理日志路径."""
    for path in [LOG_PATH, TEST_SOLUTION_PATH]:
        Path(path).mkdir(parents=True, exist_ok=True)

    for log_path in [MEMORY_DNN_LOG_PATH, SIMPLEX_LOG_PATH]:
        log_file = Path(log_path)
        if log_file.exists():
            log_file.unlink()


def loadPreprocessedData():
    """Load preprocessed data from specified paths."""
    max_values = np.load(PRE_DATA_PATH.joinpath("upper.npy"))
    max_bandwidth = np.load(PRE_DATA_PATH.joinpath("max_bandwidths.npy"))
    mapping = np.load(PRE_DATA_PATH.joinpath("node_cache_matrix.npy"))
    cost = np.load(PRE_DATA_PATH.joinpath("cost.npy"))
    node_costs = np.load(PRE_DATA_PATH.joinpath("node_costs.npy"))

    with open(PRE_DATA_PATH.joinpath("pre_data.json"), encoding="UTF-8") as f:
        cache2id = json.load(f)["cache2id"]

    return max_values, max_bandwidth, mapping, cost, node_costs, cache2id


def loadDataframeFiles():
    """加载数据文件."""
    file_names = [
        "覆盖可用cache组_v0.2.csv",
        "cache组信息_v0.2.csv",
        "节点信息_v0.2.csv",
        "覆盖信息_v0.2.csv",
        "质量等级信息_v0.2.csv",
    ]
    return {
        "coverage_cache_group_info": pd.read_csv(DATA_PATH.joinpath(file_names[0])),
        "cache_group_info": pd.read_csv(DATA_PATH.joinpath(file_names[1])),
        "node_info": pd.read_csv(DATA_PATH.joinpath(file_names[2])),
        "coverage_info": pd.read_csv(DATA_PATH.joinpath(file_names[3])),
        "quality_level_info": pd.read_csv(DATA_PATH.joinpath(file_names[4])),
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
    progress,
):
    droo_output = np.array(model.decode(droo_input, N, "OP"))
    droo_connect = droo_output - droo_input[:, np.newaxis, :] + 1
    solutions = compute_solutions(
        R,
        droo_output,
        droo_connect,
        max_values,
        max_bandwidth,
        mappings,
        cost,
        progress,
    )

    if solutions.shape[0] == 0:
        return 0, None

    _, best_solution = get_best_solution(solutions, mappings, node_cost, progress)
    return solutions.shape[0], best_solution


def getRequestsAndConnectivity(user_list, cache2id, dataframes, progress):
    users = len(user_list)
    connectivity_matrix = np.zeros((users, CACHES), dtype=bool)
    valid_users = []
    user_reqs = np.zeros(users, dtype=int)
    user_req_progress = progress.add_task("处理用户需求", total=users)

    # 使用多进程处理用户连通性
    def collect_result(result):
        i, req, connectivity = result
        user_reqs[i] = req
        connectivity_matrix[i, :] = connectivity
        valid_users.append(i)
        progress.update(user_req_progress, advance=1)

    for i, user in enumerate(user_list):
        args = (i, user, dataframes, cache2id, CACHES)
        result = processUser(args)
        collect_result(result)
    return user_reqs[valid_users], connectivity_matrix[valid_users, :]


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
    progress,
):
    # 神经网络解码和方案计算
    solution_nums = 0
    best_solution = None
    retry = math.floor(math.log2(CACHES / N)) + 1
    current = 0
    while current <= retry:
        current += 1
        progress.console.print(f"尝试第{current}次", style="bold yellow")
        solution_nums, best_solution = drooPulpPipeline(
            R,
            connectivity_matrix,
            2 ** (current - 1) * N if 2 ** (current - 1) * N < CACHES else CACHES,
            model,
            max_values,
            max_bandwidth,
            mappings,
            cost,
            node_cost,
            progress,
        )
        if solution_nums > 4:
            break
        progress.console.print("本轮未找到足够方案", style="bold yellow")
    else:
        if solution_nums == 0:
            progress.console.print("未找到任何方案", style="bold red")
        return
    progress.console.print(
        f"找到{solution_nums}个方案，得到最优解，已保存", style="bold green"
    )
    return best_solution


def infer_csv_pipeline(test_csv: Path, timepoint: int = 167, N=16):
    # 初始化控制台和模型
    console = Console()
    MemoryDNN_Net = MemoryDNN(**PARAMS)
    MemoryDNN_Net.load_state_dict(MODEL_SAVE_PATH.joinpath("latest.pth"))

    # 文件夹准备和数据加载
    prepareDirectories()
    max_values, max_bandwidth, mappings, cost, node_cost, cache2id = (
        loadPreprocessedData()
    )
    dataframes = loadDataframeFiles()

    # 加载用户需求
    df = pd.read_csv(test_csv)
    df.rename(columns={"6月用户带宽数据": "带宽数据"}, inplace=True)
    df = df[df["时间点"] == timepoint]
    # 初始化进度条和连接矩阵
    with Progress(console=console) as progress:
        user_list = []
        for row in df.itertuples(index=False):
            user_list.append(
                User_Req(
                    province=row.省份,
                    operator=row.运营商,
                    coverage_name=row.覆盖名,
                    ip_type=row.IP类型,
                    reqs=row.带宽数据,
                )
            )
        user_reqs, connectivity_matrix = getRequestsAndConnectivity(
            user_list, cache2id, dataframes, progress
        )
        if user_reqs.shape[0] == 0:
            progress.console.print("无有效用户需求", style="bold red")
            return
        best_solution: np.ndarray | None = inferWithRetry(
            R=user_reqs,
            connectivity_matrix=connectivity_matrix,
            N=N,
            model=MemoryDNN_Net,
            max_values=max_values,
            max_bandwidth=max_bandwidth,
            mappings=mappings,
            cost=cost,
            node_cost=node_cost,
            progress=progress,
        )
        if best_solution is not None:
            np.save(
                TEST_SOLUTION_PATH.joinpath(f"{test_csv.stem}_{timepoint}.npy"),
                best_solution,
            )


def random_test():
    # 获取所有可用文件
    all_files_5 = list(PRE_DATA_PATH.joinpath("5_csv_cleaned").iterdir())
    all_files_6 = list(PRE_DATA_PATH.joinpath("6_csv_cleaned").iterdir())
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
            infer_csv_pipeline(random_date_csv, timepoint, N=16)


if __name__ == "__main__":
    # infer_csv_pipeline(test_csv=Path("data/pre_data/5_csv_cleaned/2023-05-01.csv"), timepoint=3, N=16)
    random_test()
