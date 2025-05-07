import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from models import UserReq
from utils.config import (
    DATA_PATH,
    INPUT_DATA_PATH,
    MONTH_SUFFIX_CLEANED,
    PRE_DATA_PATH,
)
from utils.logger import logger


def process_timepoint(args):
    (
        timepoint,
        user_bandwidth,
        caches,
        coverage_cache_group_info,
        cache_group_info,
        node_info,
        coverage_info,
        quality_level_info,
        cache2id,
    ) = args
    time_user_bandwidth = user_bandwidth[user_bandwidth["时间点"] == timepoint]
    users = time_user_bandwidth.shape[0]
    time_connectivity_matrix = np.zeros((users, caches), dtype=bool)
    # 记录有效用户的索引
    vaild_users = []
    UserReqs = np.zeros(users, dtype=int)

    for i, row in enumerate(time_user_bandwidth.itertuples(index=False)):
        req, connectivity = UserReq(
            province=row.省份,
            operator=row.运营商,
            coverage_name=row.覆盖名,
            ip_type=row.IP类型,
            reqs=row.带宽数据,
        ).get_connectivity(
            coverage_cache_group_info,
            cache_group_info,
            node_info,
            coverage_info,
            quality_level_info,
            cache2id,
            caches,
        )
        if req == 0:
            continue
        time_connectivity_matrix[i, :] = connectivity
        UserReqs[i] = req
        vaild_users.append(i)
    time_connectivity_matrix = time_connectivity_matrix[vaild_users, :]
    vaild_timepoint_user_bandwidth = UserReqs[vaild_users]
    return timepoint, time_connectivity_matrix, vaild_timepoint_user_bandwidth


def preparing_for_droo(
    user_bandwidth: pd.DataFrame,
    save_path: Path,
    caches: int,
    coverage_cache_group_info: pd.DataFrame,
    cache_group_info: pd.DataFrame,
    node_info: pd.DataFrame,
    coverage_info: pd.DataFrame,
    quality_level_info: pd.DataFrame,
    cache2id: dict,
) -> None:
    timepoints = user_bandwidth["时间点"].nunique()
    logger.info(f"处理时间点: {timepoints}")

    def update_timepoint(result):
        timepoint, time_connectivity_matrix, timepoint_user_bandwidth = result
        if time_connectivity_matrix.shape[0] != 0:
            savemat(
                save_path.joinpath(f"{timepoint + 1}.mat"),
                {
                    "connectivity": time_connectivity_matrix,
                    "requests": timepoint_user_bandwidth,
                },
                do_compression=True,
            )

    with Pool() as pool:
        args = [
            (
                timepoint,
                user_bandwidth,
                caches,
                coverage_cache_group_info,
                cache_group_info,
                node_info,
                coverage_info,
                quality_level_info,
                cache2id,
            )
            for timepoint in range(timepoints)
        ]
        for arg in args:
            pool.apply_async(process_timepoint, args=(arg,), callback=update_timepoint)
        pool.close()
        pool.join()


if __name__ == "__main__":
    # 加载必要的 JSON 和 CSV 数据
    with open(f"{PRE_DATA_PATH}/pre_data.json", encoding="UTF-8") as f:
        cache2id = json.load(f)["cache2id"]
    caches = len(cache2id)  # 计算 cache 组数量

    # 加载 CSV 数据表格
    coverage_cache_group_info = pd.read_csv(f"{DATA_PATH}/覆盖可用cache组_v0.2.csv")
    cache_group_info = pd.read_csv(f"{DATA_PATH}/cache组信息_v0.2.csv")
    node_info = pd.read_csv(f"{DATA_PATH}/节点信息_v0.2.csv")
    coverage_info = pd.read_csv(f"{DATA_PATH}/覆盖信息_v0.2.csv")
    quality_level_info = pd.read_csv(f"{DATA_PATH}/质量等级信息_v0.2.csv")

    # 获取月份文件数量（预处理数据文件夹中的文件数量）
    months = sum(
        1
        for file in Path(PRE_DATA_PATH).iterdir()
        if file.name.endswith(MONTH_SUFFIX_CLEANED)
    )

    # 处理每个月的文件夹
    for file_path in sorted(list(PRE_DATA_PATH.iterdir())):
        if file_path.name.endswith(MONTH_SUFFIX_CLEANED):
            # 获取月份名称，例如"2024_06"格式
            month: str = file_path.name.split("_")[0]
            logger.info(f"处理月份: {month}")
            # 确保每个月份对应的输出目录存在
            month_dir = INPUT_DATA_PATH.joinpath(month)
            month_dir.mkdir(parents=True, exist_ok=True)

            # 获取每个月文件夹中的日期文件总数
            total_dates = sum(1 for _ in PRE_DATA_PATH.joinpath(file_path).iterdir())

            # 处理每个日期的文件
            for daily_file in sorted(list(PRE_DATA_PATH.joinpath(file_path).iterdir())):
                # 获取日期名称，例如"2024_06_01"格式
                date: str = daily_file.name.split(".")[0]
                logger.info(f"处理日期: {date}")

                # 设置每日输出路径，确保对应的文件夹存在
                daily_dir: Path = month_dir.joinpath(date)
                daily_dir.mkdir(parents=True, exist_ok=True)

                # 读取用户带宽数据 CSV 文件
                user_bandwidth_path = (
                    Path(PRE_DATA_PATH).joinpath(file_path).joinpath(f"{date}.csv")
                )
                user_bandwidth = pd.read_csv(user_bandwidth_path, header=0)
                user_bandwidth.rename(
                    columns={"6月用户带宽数据": "带宽数据"}, inplace=True
                )

                # 调用函数处理每日数据，并保存结果
                preparing_for_droo(
                    user_bandwidth,
                    daily_dir,
                    caches,
                    coverage_cache_group_info,
                    cache_group_info,
                    node_info,
                    coverage_info,
                    quality_level_info,
                    cache2id,
                )
