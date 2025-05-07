import numpy as np
import pandas as pd


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
        cache_group = UserReq.get_cache_group(
            self.operator, self.coverage_name, self.ip_type, coverage_cache_group_info
        )
        node_name = UserReq.get_node_name(cache_group, cache_group_info)
        connect_node_info = UserReq.get_node_info(node_name, node_info)

        # 过滤节点信息
        quality_level = UserReq.get_quality_level(self.coverage_name, coverage_info)
        connect_node_info = UserReq.filter_quality_level(
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
