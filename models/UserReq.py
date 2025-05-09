from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from optimization.SharedMemory import SharedMemManager
from utils.config import CACHES
from utils.logger import logger


class UserReq:
    def __init__(
        self, province: str, operator: str, coverage_name: str, ip_type: int, reqs: int
    ):
        self.province = province
        self.operator = operator
        self.coverage_name = coverage_name
        self.ip_type = ip_type
        self.reqs = reqs

    @staticmethod
    def get_user_req(
        user_bandwidth: pd.DataFrame,
        operator: str,
        coverage_name: str,
        ip_type: int,
        user_province: str,
        timepoint: str,
    ) -> pd.DataFrame:
        return user_bandwidth[
            (user_bandwidth["运营商"] == operator)
            & (user_bandwidth["覆盖名"] == coverage_name)
            & (user_bandwidth["IP类型"] == ip_type)
            & (user_bandwidth["省份"] == user_province)
            & (user_bandwidth["时间点"] == timepoint)
        ]

    @staticmethod
    def get_cache_group(
        operator: str,
        coverage_name: str,
        ip_type: int,
        coverage_cache_group_info: pd.DataFrame,
    ) -> pd.DataFrame:
        return coverage_cache_group_info[
            (coverage_cache_group_info["运营商"] == operator)
            & (coverage_cache_group_info["覆盖名"] == coverage_name)
            & (coverage_cache_group_info["IP类型"] == ip_type)
        ]

    @staticmethod
    def get_node_name(
        cache_group: pd.DataFrame, cache_group_info: pd.DataFrame
    ) -> pd.DataFrame:
        available_cache_groups = cache_group["可用cache组"].unique()
        return cache_group_info[
            cache_group_info["cache组"].isin(available_cache_groups)
        ][["cache组", "节点名"]]

    @staticmethod
    def get_node_info(node_name: pd.DataFrame, node_info: pd.DataFrame) -> pd.DataFrame:
        node_names = node_name["节点名"].unique()
        return node_info[node_info["节点名"].isin(node_names)].merge(
            node_name, on="节点名"
        )

    @staticmethod
    def get_quality_level(coverage_name: str, coverage_info: pd.DataFrame) -> int:
        return coverage_info[coverage_info["覆盖"] == coverage_name][
            "质量等级限制"
        ].values[0]

    @staticmethod
    def filter_quality_level(
        node_info: pd.DataFrame,
        quality_level: int,
        operator: str,
        user_province: str,
        quality_level_info: pd.DataFrame,
    ) -> pd.DataFrame:
        quality_level_info_filtered = quality_level_info[
            (quality_level_info["运营商"] == operator)
            & (quality_level_info["用户省份"] == user_province)
        ]
        province_quality_level = node_info.merge(
            quality_level_info_filtered, left_on="省份", right_on="资源省份", how="left"
        )
        return province_quality_level[
            province_quality_level["质量等级"] <= quality_level
        ][["cache组", "节点名", "质量等级"]].drop_duplicates()

    def get_connectivity(
        self,
    ) -> np.ndarray:
        """计算用户请求的连接性向量"""
        if self.reqs == 0:
            return np.array([])

        coverage_cache_group_info: pd.DataFrame = SharedMemManager.get_by_name(
            "coverage_cache_group_info"
        )
        cache_group_info: pd.DataFrame = SharedMemManager.get_by_name(
            "cache_group_info"
        )
        node_info: pd.DataFrame = SharedMemManager.get_by_name("node_info")
        coverage_info: pd.DataFrame = SharedMemManager.get_by_name("coverage_info")
        quality_level_info: pd.DataFrame = SharedMemManager.get_by_name(
            "quality_level_info"
        )
        cache2id: Dict = SharedMemManager.get_by_name("cache2id")

        # assert type
        assert isinstance(coverage_cache_group_info, pd.DataFrame), (
            f"coverage_cache_group_info should be a DataFrame, but got {type(coverage_cache_group_info)}"
        )
        assert isinstance(cache_group_info, pd.DataFrame), (
            f"cache_group_info should be a DataFrame, but got {type(cache_group_info)}"
        )
        assert isinstance(node_info, pd.DataFrame), (
            f"node_info should be a DataFrame, but got {type(node_info)}"
        )
        assert isinstance(coverage_info, pd.DataFrame), (
            f"coverage_info should be a DataFrame, but got {type(coverage_info)}"
        )
        assert isinstance(quality_level_info, pd.DataFrame), (
            f"quality_level_info should be a DataFrame, but got {type(quality_level_info)}"
        )
        assert isinstance(cache2id, dict), (
            f"cache2id should be a dictionary, but got {type(cache2id)}"
        )

        caches: int = len(cache2id)

        # 获取 cache group 和 node 信息
        cache_group: pd.DataFrame = UserReq.get_cache_group(
            self.operator, self.coverage_name, self.ip_type, coverage_cache_group_info
        )
        node_name: pd.DataFrame = UserReq.get_node_name(cache_group, cache_group_info)
        connect_node_info: pd.DataFrame = UserReq.get_node_info(node_name, node_info)

        # 过滤节点信息
        quality_level: int = UserReq.get_quality_level(
            self.coverage_name, coverage_info
        )
        connect_node_info = UserReq.filter_quality_level(
            connect_node_info,
            quality_level,
            self.operator,
            self.province,
            quality_level_info,
        )

        # 计算连接性向量
        connectivity_vector: np.ndarray = np.zeros(caches, dtype=bool)

        cache_indices: pd.Series = (
            connect_node_info["cache组"].map(cache2id).dropna().astype(int) - 1
        )
        connectivity_vector[cache_indices] = True

        return connectivity_vector if connectivity_vector.any() else np.array([])

    @staticmethod
    def from_row(row) -> "UserReq":
        return UserReq(
            province=str(row.省份),
            operator=str(row.运营商),
            coverage_name=str(row.覆盖名),
            ip_type=int(float(row.IP类型))
            if isinstance(row.IP类型, (str, float))
            else int(str(row.IP类型)),
            reqs=int(row.带宽数据)
            if isinstance(row.带宽数据, (str, float, int))
            else 0,
        )

    @staticmethod
    def getRequestsAndConnectivity(
        user_list: List["UserReq"],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get user requests and connectivity matrix."""

        users = len(user_list)
        connectivity_matrix = np.zeros((users, CACHES), dtype=bool)
        UserReqs = np.zeros(users, dtype=int)
        logger.info("Start processing user requests and connectivity matrix...")

        for i, user in enumerate(user_list):
            if user.reqs > 0:
                connectivity = user.get_connectivity()
                if connectivity.size == 0:
                    logger.debug(f"User {i} has no connectivity")
                    continue
                UserReqs[i] = user.reqs
                connectivity_matrix[i, :] = connectivity

        return UserReqs, connectivity_matrix
