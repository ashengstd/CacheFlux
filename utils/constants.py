import pathlib

# 一些常量
CACHES = 116  # 这个是缓存的数量，也是网络输入的维度
N = 16  # 这个是网络输出的方案的数量
PEEK_PERIOD = [0, 288]  # 这个是高峰期的时间段
MAX_SAMPLES_PER_DAY = 40  # 这个是每天的最大样本数量

# 本地路径
CURRENT_PATH = pathlib.Path.cwd()
DATA_PATH = CURRENT_PATH.joinpath("data/质量约束下的成本调度数据v0.2版")
PRE_DATA_PATH = CURRENT_PATH.joinpath("data/pre_data")
INPUT_DATA_PATH = CURRENT_PATH.joinpath("data/droo_mat/droo_input_data")
GLOBAL_PATH = CURRENT_PATH.joinpath("data/simplex")
BEST_SOLUTION_PATH = GLOBAL_PATH.joinpath("best_solutions")
MODEL_SAVE_PATH = CURRENT_PATH.joinpath("ckpts")
TEST_SOLUTION_PATH = GLOBAL_PATH.joinpath("test_solutions")

# 日志文件
PULP_LOG_ENABLE = True
MEMORY_DNN_LOG_ENABLE = True
LOG_PATH = CURRENT_PATH.joinpath("logs/")
SIMPLEX_LOG_PATH = LOG_PATH.joinpath("simplex.log")
MEMORY_DNN_LOG_PATH = LOG_PATH.joinpath("memory_dnn.log")

# 特殊字符串
MONTH_SUFFIX_CLEANED = "_csv_cleaned"

# 网络参数
PARAMS = {
    "network_architecture": [CACHES, 120, 80, CACHES],
    "learning_rate": 0.007,
    "training_interval": 1,
    "batch_size": 128,
    "memory_size": 1024,
    "log_file": MEMORY_DNN_LOG_PATH,
    "log_enable": MEMORY_DNN_LOG_ENABLE,
}
THREADS = 8
