from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.io import savemat

from models import UserReq
from optimization.SharedMemory import SharedMemManager
from utils.config import (
    INPUT_DATA_PATH,
    REQ_CSV_CSV_PATH,
)
from utils.func import loadInfoData
from utils.logger import logger


def process_timepoint(
    time_user_bandwidths: pd.DataFrame,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Process a single timepoint of user bandwidth data."""
    cache2id = SharedMemManager.get_by_name("cache2id")
    assert isinstance(cache2id, dict), "cache2id should be a dictionary"
    timepoint = time_user_bandwidths["时间点"].values[0]
    caches = len(cache2id)
    if time_user_bandwidths.shape[0] == 0:
        return timepoint, np.zeros((0, caches), dtype=bool), np.zeros(0, dtype=int)
    users = time_user_bandwidths.shape[0]
    time_connectivity_matrix = np.zeros((users, caches), dtype=bool)
    UserReqs = np.zeros(users, dtype=int)

    for i, row in enumerate(time_user_bandwidths.itertuples(index=False)):
        connectivity = UserReq.from_row(row=row).get_connectivity()
        if connectivity.size == 0:
            logger.debug(f"User {i} has no connectivity")
            continue
        time_connectivity_matrix[i, :] = connectivity
        UserReqs[i] = row.带宽数据
    time_connectivity_matrix = time_connectivity_matrix
    print(time_connectivity_matrix.any(), time_connectivity_matrix.sum())
    return timepoint, time_connectivity_matrix, UserReqs


def preparing_for_droo(
    user_bandwidth: pd.DataFrame,
    save_path: Path,
) -> None:
    timepoints = user_bandwidth["时间点"].nunique()
    logger.info(f"All timepoints needed to process: {timepoints}")

    for timepoint in range(timepoints):
        timepoint_user_bandwidths = user_bandwidth[
            user_bandwidth["时间点"] == timepoint
        ]
        user_list = [
            UserReq.from_row(row=row) for row in timepoint_user_bandwidths.itertuples()
        ]
        user_list = [user for user in user_list if user.reqs > 0]
        timepoint_user_bandwidth, time_connectivity_matrix = (
            UserReq.getRequestsAndConnectivity(
                user_list=user_list,
            )
        )

        if time_connectivity_matrix.shape[0] != 0:
            logger.info(f"Timepoint {timepoint + 1} processed successfully, ")
            savemat(
                save_path.joinpath(f"{timepoint + 1}.mat"),
                {
                    "connectivity": time_connectivity_matrix,
                    "requests": timepoint_user_bandwidth,
                },
                do_compression=True,
            )

    logger.info("All timepoints processed successfully.")


if __name__ == "__main__":
    loadInfoData()
    # 获取月份文件数量（预处理数据文件夹中的文件数量）
    months = sum(1 for _ in Path(REQ_CSV_CSV_PATH).iterdir())

    # 处理每个月的文件夹
    for month_dir in sorted(list(REQ_CSV_CSV_PATH.iterdir())):
        # 获取月份名称，例如"2024_06"格式
        month: str = month_dir.name
        logger.info(f"Processing Month: {month}")
        # 确保每个月份对应的输出目录存在
        month_dir = INPUT_DATA_PATH.joinpath(month)
        month_dir.mkdir(parents=True, exist_ok=True)

        # 处理每个日期的文件
        for daily_file in sorted(list(REQ_CSV_CSV_PATH.joinpath(month).iterdir())):
            # 获取日期名称，例如"2024_06_01"格式
            date: str = daily_file.name.split(".")[0]
            logger.info(f"Processing Date: {date}")

            # 设置每日输出路径，确保对应的文件夹存在
            daily_dir: Path = month_dir.joinpath(date)
            daily_dir.mkdir(parents=True, exist_ok=True)

            # 读取用户带宽数据 CSV 文件
            user_bandwidth_path = (
                Path(REQ_CSV_CSV_PATH).joinpath(month).joinpath(f"{date}.csv")
            )
            user_bandwidth = pd.read_csv(user_bandwidth_path, header=0)
            user_bandwidth.rename(columns={"6月用户带宽数据": "带宽数据"}, inplace=True)

            # 调用函数处理每日数据，并保存结果
            preparing_for_droo(
                user_bandwidth,
                daily_dir,
            )
