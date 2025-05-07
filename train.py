import argparse
import random
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np
import pandas as pd
import scipy.io as sio
from rich.console import Console
from rich.progress import Progress

from models import MemoryConfig, plMemoryDNN
from optimization.solutions import compute_solutions, get_best_solution
from utils.constants import (
    BEST_SOLUTION_PATH,
    INPUT_DATA_PATH,
    LOG_PATH,
    MAX_SAMPLES_PER_DAY,
    MEMORY_DNN_LOG_PATH,
    MODEL_SAVE_PATH,
    MONTH_SUFFIX_CLEANED,
    PEEK_PERIOD,
    PRE_DATA_PATH,
    PULP_LOG_ENABLE,
    SIMPLEX_LOG_PATH,
    N,
)


def setup_directories():
    """Create necessary directories and clean up log files."""
    Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
    if Path(MEMORY_DNN_LOG_PATH).exists():
        Path(MEMORY_DNN_LOG_PATH).unlink()
    if Path(SIMPLEX_LOG_PATH).exists():
        Path(SIMPLEX_LOG_PATH).unlink()
    Path(BEST_SOLUTION_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)


def load_preprocessed_data():
    """Load preprocessed data from specified paths."""
    max_values = np.load(PRE_DATA_PATH.joinpath("upper.npy"))
    max_bandwidth = np.load(PRE_DATA_PATH.joinpath("max_bandwidths.npy"))
    mapping = np.load(PRE_DATA_PATH.joinpath("node_cache_matrix.npy"))
    cost = np.load(PRE_DATA_PATH.joinpath("cost.npy"))
    node_costs = np.load(PRE_DATA_PATH.joinpath("node_costs.npy"))
    return max_values, max_bandwidth, mapping, cost, node_costs


def sample_files(
    peek_period: List[Path], normal_period: List[Path], max_files: int = 40
) -> List[Path]:
    sampled_files: List[Path] = []
    # 抽取 50% 概率选择高峰期范围内的文件
    while len(sampled_files) < max_files and (peek_period or normal_period):
        if random.random() < 0.5 and peek_period:  # 50%概率从高峰期中选择
            selected_file = random.choice(peek_period)
            sampled_files.append(selected_file)
            peek_period.remove(selected_file)
        elif normal_period:  # 其余情况下从其他文件中选择
            selected_file = random.choice(normal_period)
            sampled_files.append(selected_file)
            normal_period.remove(selected_file)
    return sampled_files


def train_droo_net_pulp_pipline(
    timepoint: int,
    droo_input_path: Path,
    MemoryDNN_Net: plMemoryDNN,
    progress: Progress,
    max_values: np.ndarray,
    max_bandwidth: np.ndarray,
    mappings: np.ndarray,
    cost: np.ndarray,
    node_costs: np.ndarray,
) -> tuple[int, np.ndarray | None]:
    """Main pipeline for training the DROO network using the simplex method."""
    droo_data: Mapping[Any, Any] = sio.loadmat(
        droo_input_path.joinpath(f"{timepoint + 1}.mat")
    )
    # 提取第 i 个时间点的 users*caches 矩阵
    timepoint_data: np.ndarray = droo_data["connectivity"]
    # 调用 mem.decode() 得到 droo_output，返回神经网络的输出
    droo_output: np.ndarray = np.array(
        MemoryDNN_Net.decode(timepoint_data, N), dtype=np.bool_
    )
    # 计算需求矩阵
    R: np.ndarray = droo_data["requests"].squeeze()
    # droo_connect用于保存奖励函数的差值
    droo_connect: np.ndarray = droo_output - timepoint_data[:, np.newaxis, :] + 1
    solutions: np.ndarray = compute_solutions(
        R,
        droo_output,
        droo_connect,
        max_values,
        max_bandwidth,
        mappings,
        cost,
        progress,
    )
    if solutions.size == 0:
        return 0, None
    # 获得最优方案
    best_index: int
    best_solution: np.ndarray
    best_index, best_solution = get_best_solution(
        solutions, mappings, node_costs, progress
    )
    # 每5个时间点训练一次神经网络
    MemoryDNN_Net.encode(timepoint_data, droo_output[:, best_index, :])
    return solutions.shape[0], best_solution


if __name__ == "__main__":
    # 初始化控制台
    console = Console()
    parser = argparse.ArgumentParser(description="Train the DROO network")
    parser.add_argument(
        "--peek_period_enabled", action="store_true", help="Enable peek period"
    )
    args = parser.parse_args()

    PL_PARAMS = MemoryConfig(network_architecture=[116, 120, 80, 116])
    # 从字典中解包参数并创建模型
    MemoryDNN_Net = plMemoryDNN(PL_PARAMS)

    # 检查文件夹
    setup_directories()

    # 加载预处理数据
    max_values: np.ndarray
    max_bandwidth: np.ndarray
    mappings: np.ndarray
    cost: np.ndarray
    node_costs: np.ndarray
    max_values, max_bandwidth, mappings, cost, node_costs = load_preprocessed_data()

    # 创建进度条
    with Progress(console=console) as progress:
        months_task = progress.add_task(
            "处理月份", total=sum(1 for _ in INPUT_DATA_PATH.iterdir())
        )
        epoch = 0
        # 调用单纯形法
        for monthly_dir in INPUT_DATA_PATH.iterdir():
            month = monthly_dir.name
            progress.update(months_task, description=f"处理月份：{month}")
            daily_task = progress.add_task(
                "处理日期",
                total=sum(1 for _ in INPUT_DATA_PATH.joinpath(month).iterdir()),
            )

            for daily_dir in INPUT_DATA_PATH.joinpath(month).iterdir():
                date: str = daily_dir.name.split(".")[0]
                if PULP_LOG_ENABLE:
                    with open(SIMPLEX_LOG_PATH, "a") as log_file:
                        log_file.write(f"Date: {date}:\n")
                progress.update(daily_task, description=f"处理日期：{date}")
                droo_input_path: Path = INPUT_DATA_PATH.joinpath(month).joinpath(date)
                user_bandwidth: pd.DataFrame = pd.read_csv(
                    PRE_DATA_PATH.joinpath(f"{month}{MONTH_SUFFIX_CLEANED}").joinpath(
                        f"{date}.csv"
                    ),
                    header=0,
                )
                timepoints = user_bandwidth["时间点"].nunique()
                timepoint_task = progress.add_task("处理时间点", total=100)
                best_solution_save_path = BEST_SOLUTION_PATH.joinpath(date)
                best_solution_save_path.mkdir(parents=True, exist_ok=True)

                # 抽样时间点
                timepoint_files = sorted(
                    list(INPUT_DATA_PATH.joinpath(month).joinpath(date).iterdir()),
                    key=lambda x: int(x.stem),
                )
                if not args.peek_period_enabled:
                    peek_period = [
                        file
                        for file in timepoint_files
                        if PEEK_PERIOD[0] <= int(file.stem) <= PEEK_PERIOD[1]
                    ]
                    normal_period = [
                        file for file in timepoint_files if file not in peek_period
                    ]
                    sampled_files = sample_files(
                        peek_period, normal_period, MAX_SAMPLES_PER_DAY
                    )
                else:
                    sampled_files = timepoint_files
                progress.update(timepoint_task, total=len(sampled_files))
                for timepoint_file in sampled_files:
                    epoch += 1
                    timepoint = int(timepoint_file.name.split(".")[0]) - 1
                    if PULP_LOG_ENABLE:
                        with open(SIMPLEX_LOG_PATH, "a") as log_file:
                            log_file.write(f"Timepoint {timepoint + 1}:\n")
                    progress.update(
                        timepoint_task, description=f"处理时间点：{timepoint + 1}"
                    )
                    solutions_nums: int | None
                    best_solution: np.ndarray | None
                    solutions_nums, best_solution = train_droo_net_pulp_pipline(
                        timepoint=timepoint,
                        droo_input_path=droo_input_path,
                        MemoryDNN_Net=MemoryDNN_Net,
                        progress=progress,
                        max_values=max_values,
                        max_bandwidth=max_bandwidth,
                        mappings=mappings,
                        cost=cost,
                        node_costs=node_costs,
                    )
                    if solutions_nums == 0:
                        progress.console.print("No solutions found!", style="bold red")
                    else:
                        progress.console.print(
                            f"find {solutions_nums} solutions", style="bold green"
                        )
                        if best_solution is not None:
                            np.save(
                                best_solution_save_path.joinpath(
                                    f"{timepoint + 1}.npy"
                                ),
                                best_solution,
                            )

                    if epoch % 5 == 0:
                        MemoryDNN_Net.save_model(MODEL_SAVE_PATH.joinpath("latest"))
                    progress.advance(timepoint_task)
                # 保存结果
                progress.remove_task(timepoint_task)
                progress.advance(daily_task)
                MemoryDNN_Net.save_model(MODEL_SAVE_PATH.joinpath(month))
            progress.advance(months_task)
        progress.remove_task(months_task)
        MemoryDNN_Net.save_model(MODEL_SAVE_PATH.joinpath("best"))
        console.print("训练完成!")
