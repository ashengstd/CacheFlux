import numpy as np
import pandas as pd

from utils.DataFrameUtils import (
    filter_quality_level,
    get_cache_group,
    get_node_info,
    get_node_name,
    get_quality_level,
)


class UserReq:
    def __init__(
        self, province: str, operator: str, coverage_name: str, ip_type: int, reqs: int
    ):
        self.province = province
        self.operator = operator
        self.coverage_name = coverage_name
        self.ip_type = ip_type
        self.reqs = reqs

    def get_connectivity(
        self,
        coverage_cache_group_info: pd.DataFrame,
        cache_group_info: pd.DataFrame,
        node_info: pd.DataFrame,
        coverage_info: pd.DataFrame,
        quality_level_info: pd.DataFrame,
        cache2id: dict[str, int],
        caches: int,
    ) -> tuple[int, np.ndarray | None]:
        """计算用户请求的连接性向量"""
        if self.reqs == 0:
            return 0, None

        # 获取 cache group 和 node 信息
        cache_group = get_cache_group(
            self.operator, self.coverage_name, self.ip_type, coverage_cache_group_info
        )
        node_name = get_node_name(cache_group, cache_group_info)
        connect_node_info = get_node_info(node_name, node_info)

        # 过滤节点信息
        quality_level = get_quality_level(self.coverage_name, coverage_info)
        connect_node_info = filter_quality_level(
            connect_node_info,
            quality_level,
            self.operator,
            self.province,
            quality_level_info,
        )

        # 计算连接性向量
        connectivity_vector = np.zeros(caches, dtype=bool)
        cache_indices = (
            connect_node_info["cache组"].map(cache2id).dropna().astype(int) - 1
        )
        connectivity_vector[cache_indices] = True

        return (
            (self.reqs, connectivity_vector) if connectivity_vector.any() else (0, None)
        )
