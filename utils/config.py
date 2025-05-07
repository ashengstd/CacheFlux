import pathlib
import tomllib

CURRENT_PATH = pathlib.Path(__file__).parent.parent
config_path = CURRENT_PATH / "config.toml"

with open(config_path, "rb") as f:
    config = tomllib.load(f)

# 常量
CACHES = config["constants"]["CACHES"]
N = config["constants"]["N"]
PEEK_PERIOD = config["constants"]["PEEK_PERIOD"]
MAX_SAMPLES_PER_DAY = config["constants"]["MAX_SAMPLES_PER_DAY"]

# 路径
DATA_PATH = CURRENT_PATH / config["paths"]["data_root"]
PRE_DATA_PATH = CURRENT_PATH / config["paths"]["pre_data"]
INPUT_DATA_PATH = CURRENT_PATH / config["paths"]["input_data"]
GLOBAL_PATH = CURRENT_PATH / config["paths"]["global"]
BEST_SOLUTION_PATH = CURRENT_PATH / config["paths"]["best_solution"]
TEST_SOLUTION_PATH = CURRENT_PATH / config["paths"]["test_solution"]
MODEL_SAVE_PATH = CURRENT_PATH / config["paths"]["model_ckpt"]
LOG_PATH = CURRENT_PATH / config["paths"]["log_dir"]
LP_LOG_PATH = CURRENT_PATH / config["paths"]["lp_log"]
MEMORY_DNN_LOG_PATH = CURRENT_PATH / config["paths"]["memory_dnn_log"]

# 日志与特殊配置
PULP_LOG_ENABLE = config["logs"]["pulp_log_enable"]
MEMORY_DNN_LOG_ENABLE = config["logs"]["memory_dnn_log_enable"]
MONTH_SUFFIX_CLEANED = config["special"]["month_suffix_cleaned"]

# 网络参数
PARAMS = {
    "network_architecture": config["params"]["network_architecture"],
    "learning_rate": config["params"]["learning_rate"],
    "training_interval": config["params"]["training_interval"],
    "batch_size": config["params"]["batch_size"],
    "memory_size": config["params"]["memory_size"],
    "log_file": MEMORY_DNN_LOG_PATH,
    "log_enable": MEMORY_DNN_LOG_ENABLE,
}
PL_PARAMS = config["params"]

# 线程数
THREADS = config["general"]["threads"]
