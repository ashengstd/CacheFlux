import argparse
import random
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np
import pandas as pd
import scipy.io as sio
from rich.console import Console

from models import MemoryConfig, plMemoryDNN
from optimization.solutions import compute_solutions, get_best_solution
from utils.config import (
    BEST_SOLUTION_PATH,
    CSV_PATH,
    INPUT_DATA_PATH,
    LOG_PATH,
    LP_LOG_PATH,
    MAX_SAMPLES_PER_DAY,
    MEMORY_DNN_LOG_PATH,
    MODEL_SAVE_PATH,
    PEEK_PERIOD,
    PL_PARAMS,
    PRE_DATA_PATH,
    PULP_LOG_ENABLE,
    N,
)
from utils.logger import logger


def setup_directories():
    """Create necessary directories and clean up log files."""
    Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
    if Path(MEMORY_DNN_LOG_PATH).exists():
        Path(MEMORY_DNN_LOG_PATH).unlink()
    if Path(LP_LOG_PATH).exists():
        Path(LP_LOG_PATH).unlink()
    Path(BEST_SOLUTION_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)


def load_preprocessed_data():
    """Load preprocessed data from specified paths."""
    info_dif = PRE_DATA_PATH.joinpath("info")
    max_values = np.load(info_dif.joinpath("upper.npy"))
    max_bandwidth = np.load(info_dif.joinpath("max_bandwidths.npy"))
    mapping = np.load(info_dif.joinpath("node_cache_matrix.npy"))
    cost = np.load(info_dif.joinpath("cost.npy"))
    node_costs = np.load(info_dif.joinpath("node_costs.npy"))
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
    )
    if solutions.size == 0:
        return 0, None
    # 获得最优方案
    best_index: int
    best_solution: np.ndarray
    best_index, _, best_solution = get_best_solution(solutions, mappings, node_costs)
    # 每5个时间点训练一次神经网络
    MemoryDNN_Net.encode(timepoint_data, droo_output[:, best_index, :])
    return solutions.shape[0], best_solution


if __name__ == "__main__":
    # 初始化控制台
    console = Console()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train the DROO network")
    parser.add_argument(
        "--peek_period_enabled",
        default=True,
        help="Enable peek period",
    )
    args = parser.parse_args()

    # 初始化模型参数
    memory_config = MemoryConfig(**PL_PARAMS)
    MemoryDNN_Net = plMemoryDNN(memory_config)

    # 创建必要目录
    setup_directories()

    # 加载预处理数据
    max_values, max_bandwidth, mappings, cost, node_costs = load_preprocessed_data()

    # 训练 DROO 网络
    logger.info("Start training DRLOP network...")

    epoch = 0
    for monthly_dir in INPUT_DATA_PATH.iterdir():
        month = monthly_dir.name
        logger.info(f"Processing Month：{month}")
        daily_dirs = list(monthly_dir.iterdir())

        for daily_dir in daily_dirs:
            date = daily_dir.name.split(".")[0]
            droo_input_path = monthly_dir / date
            logger.info(f"Processing Date：{date}")
            # 加载 CSV
            user_bandwidth = pd.read_csv(
                CSV_PATH.joinpath(f"{month}").joinpath(f"{date}.csv"),
                header=0,
            )

            # 时间点数量
            timepoints = user_bandwidth["时间点"].nunique()
            best_solution_save_path = BEST_SOLUTION_PATH / date
            best_solution_save_path.mkdir(parents=True, exist_ok=True)

            # 抽样时间点文件
            timepoint_files = sorted(
                list(droo_input_path.iterdir()), key=lambda x: int(x.stem)
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

            # 日志记录
            if PULP_LOG_ENABLE:
                with open(LP_LOG_PATH, "a") as log_file:
                    log_file.write(f"Date: {date}:\n")

            for idx, timepoint_file in enumerate(sampled_files):
                epoch += 1
                timepoint = int(timepoint_file.name.split(".")[0]) - 1

                if PULP_LOG_ENABLE:
                    with open(LP_LOG_PATH, "a") as log_file:
                        log_file.write(f"Timepoint {timepoint + 1}:\n")

                logger.info(f"Processing Timepoint：{timepoint + 1}")

                solutions_nums, best_solution = train_droo_net_pulp_pipline(
                    timepoint=timepoint,
                    droo_input_path=droo_input_path,
                    MemoryDNN_Net=MemoryDNN_Net,
                    max_values=max_values,
                    max_bandwidth=max_bandwidth,
                    mappings=mappings,
                    cost=cost,
                    node_costs=node_costs,
                )

                if solutions_nums == 0:
                    logger.error("No solutions found!")
                else:
                    logger.info(f"find {solutions_nums} solutions")
                    if best_solution is not None:
                        pass
                        np.save(
                            best_solution_save_path / f"{timepoint + 1}.npy",
                            best_solution,
                        )

                # 每5次保存一次模型
                if epoch % 5 == 0:
                    MemoryDNN_Net.save_model(MODEL_SAVE_PATH / "latest")

            # 每天结束，保存一次
            MemoryDNN_Net.save_model(MODEL_SAVE_PATH / month)

    # 最终保存
    MemoryDNN_Net.save_model(MODEL_SAVE_PATH / "best")
    logger.info("Training Done!")
