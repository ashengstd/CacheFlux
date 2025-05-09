import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from optimization.SharedMemory import SharedMemManager
from utils.config import (
    INFO_CSV_PATH,
    INFO_NPY_PATH,
)
from utils.logger import logger


def load_preprocessed_npy_data() -> None:
    """Load preprocessed data from specified paths."""
    logger.info("Loading preprocessed npy data...")
    max_values = np.load(INFO_NPY_PATH.joinpath("uppers.npy"))
    max_bandwidths = np.load(INFO_NPY_PATH.joinpath("max_bandwidths.npy"))
    mappings = np.load(INFO_NPY_PATH.joinpath("node_cache_matrix.npy"))
    costs = np.load(INFO_NPY_PATH.joinpath("costs.npy"))
    node_costs = np.load(INFO_NPY_PATH.joinpath("node_costs.npy"))
    logger.info("Preprocessed npy data loaded successfully.")
    logger.info("Creating shared memory blocks...")
    for name, array in {
        "max_values": max_values,
        "max_bandwidths": max_bandwidths,
        "mappings": mappings,
        "costs": costs,
        "node_costs": node_costs,
    }.items():
        SharedMemManager.create_block(name, array)
    logger.info(
        "Shared memory blocks created successfully for the preprocessed npy data."
    )


def loadInfoData() -> None:
    """
    Load CSV files into DataFrames and store them in shared memory.
    """
    file_names = [
        "coverage_cache_group_info.csv",
        "cache_group_info.csv",
        "node_info.csv",
        "coverage_info.csv",
        "quality_level_info.csv",
    ]
    csv_data_dict = {
        "coverage_cache_group_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[0])),
        "cache_group_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[1])),
        "node_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[2])),
        "coverage_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[3])),
        "quality_level_info": pd.read_csv(INFO_CSV_PATH.joinpath(file_names[4])),
    }
    for name, df in csv_data_dict.items():
        SharedMemManager.create_block(name, df)
    with open(INFO_NPY_PATH.joinpath("cache2id.json"), "r") as f:
        cache2id: Dict = json.load(f)
    SharedMemManager.create_block("cache2id", cache2id)


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


def handle_error(e):
    logger.error(f"Error in child process: {e}")
