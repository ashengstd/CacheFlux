from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import scipy.io as sio
from fire import Fire

from models import MemoryConfig, plMemoryDNN
from optimization.SharedMemory import SharedMemManager
from optimization.solutions import compute_solutions, get_best_solution
from utils.config import (
    INPUT_DATA_PATH,
    MAX_SAMPLES_PER_DAY,
    MODEL_SAVE_PATH,
    PEEK_PERIOD,
    PL_PARAMS,
    N,
)
from utils.func import load_preprocessed_npy_data, sample_files
from utils.logger import logger


def train_droo_net_pulp_pipline(
    timepoint: int,
    droo_input_path: Path,
    MemoryDNN_Net: plMemoryDNN,
) -> tuple[int, np.ndarray | None]:
    """Main pipeline for training the DROO network using the simplex method."""
    mat: Mapping[Any, Any] = sio.loadmat(
        droo_input_path.joinpath(f"{timepoint + 1}.mat")
    )
    # 提取第 i 个时间点的 users*caches 矩阵
    timepoint_data: np.ndarray = mat["connectivity"]
    # 调用 mem.decode() 得到 droo_output，返回神经网络的输出
    droo_output: np.ndarray = np.array(MemoryDNN_Net.decode(timepoint_data, N))
    # 计算需求矩阵
    R: np.ndarray = mat["requests"].squeeze()
    # droo_connect用于保存奖励函数的差值
    droo_connect: np.ndarray = droo_output - timepoint_data[:, np.newaxis, :] + 1
    solutions: np.ndarray = compute_solutions(R=R, I_ij=droo_output, A_ij=droo_connect)
    if solutions.size == 0:
        return 0, None
    # 获得最优方案
    best_index: int
    best_solution: np.ndarray
    best_index, _, best_solution = get_best_solution(solutions)
    # 每5个时间点训练一次神经网络
    MemoryDNN_Net.encode(timepoint_data, droo_output[:, best_index, :])
    return solutions.shape[0], best_solution


def main(
    peek_period_enabled: bool = True,
    load_model_path: Optional[Path] = None,
) -> None:
    """Main function to train the DROO network."""
    # Initialize model
    memory_config = MemoryConfig(**PL_PARAMS)
    plMemoryDNNModel = plMemoryDNN(memory_config)

    if load_model_path is not None and load_model_path.exists():
        logger.info(f"Loading model from {load_model_path}...")
        plMemoryDNNModel.load_model(load_model_path)

    # load preprocessed data
    load_preprocessed_npy_data()

    # 训练 DROO 网络
    logger.info("Start training DRLOP network...")

    training_epoch = 0
    for monthly_dir in INPUT_DATA_PATH.iterdir():
        month = monthly_dir.name
        logger.info(f"Processing Month：{month}")
        daily_dirs = list(monthly_dir.iterdir())

        for daily_dir in daily_dirs:
            date = daily_dir.name.split(".")[0]
            droo_input_path = monthly_dir / date
            logger.info(f"Processing Date：{date}")

            # 抽样时间点文件
            timepoint_files = sorted(
                list(droo_input_path.iterdir()), key=lambda x: int(x.stem)
            )
            if not peek_period_enabled:
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

            for _, timepoint_file in enumerate(sampled_files):
                timepoint = int(timepoint_file.name.split(".")[0]) - 1

                logger.info(f"Processing Timepoint：{timepoint + 1}")

                solutions_nums, _ = train_droo_net_pulp_pipline(
                    timepoint=timepoint,
                    droo_input_path=droo_input_path,
                    MemoryDNN_Net=plMemoryDNNModel,
                )

                if solutions_nums == 0:
                    logger.error("No solutions found!")
                else:
                    training_epoch += 1
                    logger.info(f"find {solutions_nums} solutions")

                # 每5次保存一次模型
                if training_epoch % 5 == 0:
                    plMemoryDNNModel.save_model(MODEL_SAVE_PATH / "latest")

            # 每天结束，保存一次
            plMemoryDNNModel.save_model(MODEL_SAVE_PATH / month)

    # 最终保存
    plMemoryDNNModel.save_model(MODEL_SAVE_PATH / "final")
    logger.info("Training Done!")
    # 清理共享内存
    SharedMemManager.cleanup()
    logger.info("Shared memory cleaned up.")


if __name__ == "__main__":
    Fire(main)
