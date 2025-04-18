import json

import numpy as np
import pandas as pd
from utils.constants import DATA_PATH, PRE_DATA_PATH

# 创建预处理数据目录
PRE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# 读取数据
df_cache_info = pd.read_csv(DATA_PATH.joinpath("cache组信息_v0.2.csv"))
df_node_info = pd.read_csv(DATA_PATH.joinpath("节点信息_v0.2.csv"))

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
node_cache_matrix = np.zeros((len(node2id), len(cache2id)), dtype=int)
for _, row in df_cache_info.iterrows():
    nodeid, cacheid = node2id[row["节点名"]], cache2id[row["cache组"]]
    node_cache_matrix[nodeid, cacheid] = 1

# 保存 node_cache_matrix 和映射关系
np.save(PRE_DATA_PATH.joinpath("node_cache_matrix.npy"), node_cache_matrix)
with open(PRE_DATA_PATH.joinpath("pre_data.json"), "w") as f:
    json.dump(
        {"node2id": node2id, "cache2id": cache2id, "cache2node": cache_to_node_ids}, f
    )
print("pre_data.json and node_cache_matrix.npy saved")

# 构建并保存 B_jk_max 矩阵
B_jk_max = (
    df_cache_info.groupby("cache组")["最大可承载带宽"]
    .apply(lambda x: x.max())
    .reindex(list(cache2id.keys()))
    .to_numpy()
)
np.save(PRE_DATA_PATH.joinpath("B_jk_max.npy"), B_jk_max)
print("B_jk_max.npy saved")

# 填充 upper 和 lower 数组
# 预处理节点信息一次，避免重复set_index操作
node_info_indexed = df_node_info.set_index("节点名")
upper = node_info_indexed.reindex(list(node2id.keys()))["跑高线"].to_numpy()
lower = node_info_indexed.reindex(list(node2id.keys()))["保底线"].to_numpy()

np.save(PRE_DATA_PATH.joinpath("upper.npy"), upper)
np.save(PRE_DATA_PATH.joinpath("lower.npy"), lower)
print("upper.npy and lower.npy saved")

# 计算并保存每个 cache 组的成本
costs = np.array(
    [
        node_info_indexed.iloc[first_node_id]["计费系数"]
        for first_node_id in [cache_to_node_ids[cache][0] for cache in cache2id.keys()]
    ]
)
np.save(PRE_DATA_PATH.joinpath("cost.npy"), costs)
print("cost.npy saved")

# 计算并保存每个 node 的成本
node_costs = node_info_indexed.reindex(list(node2id.keys()))["计费系数"].to_numpy()
np.save(PRE_DATA_PATH.joinpath("node_costs.npy"), node_costs)

# 计算并保存每个 cache 组的最大带宽
max_bandwidths = (
    df_cache_info.groupby("cache组")["最大可承载带宽"]
    .max()
    .reindex(list(cache2id.keys()))
    .to_numpy()
)
np.save(PRE_DATA_PATH.joinpath("max_bandwidths.npy"), max_bandwidths)
print("max_bandwidths.npy saved")
