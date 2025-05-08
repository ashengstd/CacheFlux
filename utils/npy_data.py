import json

import numpy as np
import pandas as pd

from utils.config import INFO_CSV_PATH, INFO_NPY_PATH
from utils.logger import logger

# 读取数据
df_cache_info = pd.read_csv(INFO_CSV_PATH.joinpath("cache_group_info.csv"))
df_node_info = pd.read_csv(INFO_CSV_PATH.joinpath("node_info.csv"))

# 获取唯一的 cache 组和节点名，并创建 ID 映射
cache2id = {cache: i for i, cache in enumerate(df_cache_info["cache组"].unique())}
node2id = {node: i for i, node in enumerate(df_cache_info["节点名"].unique())}

# 创建 cache 组对应节点 ID 的字典
cache_to_node_ids = {
    cache: df_cache_info[df_cache_info["cache组"] == cache]["节点名"]
    .map(node2id)
    .tolist()
    for cache in cache2id
}

# 创建节点与 cache 组的映射矩阵
node_cache_matrix = np.zeros((len(node2id), len(cache2id)), dtype=np.bool_)
for _, row in df_cache_info.iterrows():
    nodeid, cacheid = node2id[row["节点名"]], cache2id[row["cache组"]]
    node_cache_matrix[nodeid, cacheid] = 1

# 保存 node_cache_matrix 和映射关系
np.save(INFO_NPY_PATH.joinpath("node_cache_matrix.npy"), node_cache_matrix)
with open(INFO_NPY_PATH.joinpath("pre_data.json"), "w") as f:
    json.dump(
        {"node2id": node2id, "cache2id": cache2id, "cache2node": cache_to_node_ids}, f
    )
logger.info("pre_data.json and node_cache_matrix.npy saved")

# 构建并保存 B_jk_max 矩阵
B_jk_max = (
    df_cache_info.groupby("cache组")["最大可承载带宽"]
    .apply(lambda x: x.max())
    .reindex(list(cache2id.keys()))
    .to_numpy()
)
np.save(INFO_NPY_PATH.joinpath("B_jk_maxs.npy"), B_jk_max)
logger.info("B_jk_maxs.npy saved")

# 填充 upper 和 lower 数组
# 预处理节点信息一次，避免重复set_index操作
node_info_indexed = df_node_info.set_index("节点名")
upper = node_info_indexed.reindex(list(node2id.keys()))["跑高线"].to_numpy()
lower = node_info_indexed.reindex(list(node2id.keys()))["保底线"].to_numpy()

np.save(INFO_NPY_PATH.joinpath("uppers.npy"), upper)
np.save(INFO_NPY_PATH.joinpath("lowers.npy"), lower)
logger.info("uppers.npy and lowers.npy saved")

# 计算并保存每个 cache 组的成本
costs = np.array(
    [
        node_info_indexed.iloc[first_node_id]["计费系数"]
        for first_node_id in [cache_to_node_ids[cache][0] for cache in cache2id.keys()]
    ]
)
np.save(INFO_NPY_PATH.joinpath("costs.npy"), costs)
logger.info("costs.npy saved")

# 计算并保存每个 node 的成本
node_costs = node_info_indexed.reindex(list(node2id.keys()))["计费系数"].to_numpy()
np.save(INFO_NPY_PATH.joinpath("node_costs.npy"), node_costs)

# 计算并保存每个 cache 组的最大带宽
max_bandwidths = (
    df_cache_info.groupby("cache组")["最大可承载带宽"]
    .max()
    .reindex(list(cache2id.keys()))
    .to_numpy()
)
np.save(INFO_NPY_PATH.joinpath("max_bandwidths.npy"), max_bandwidths)
logger.info("max_bandwidths.npy saved")
