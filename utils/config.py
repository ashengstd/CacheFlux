import pathlib
import tomllib

CURRENT_PATH = pathlib.Path(__file__).parent.parent
config_path = CURRENT_PATH / "config.toml"

with open(config_path, "rb") as f:
    config = tomllib.load(f)

# Constants
CACHES = config["constants"]["CACHES"]
N = config["constants"]["N"]
PEEK_PERIOD = config["constants"]["PEEK_PERIOD"]
MAX_SAMPLES_PER_DAY = config["constants"]["MAX_SAMPLES_PER_DAY"]

# Paths to the data files and directories
CSV_DATA_PATH = CURRENT_PATH / config["paths"]["csv_data"]
INFO_DATA_PATH = CURRENT_PATH / config["paths"]["info_data"]

# Paths to the preprocessed data and directories
PRE_DATA_PATH = CURRENT_PATH / config["paths"]["preprocessed_data"]
CSV_PATH = PRE_DATA_PATH.joinpath("csv")
INFO_PATH = PRE_DATA_PATH.joinpath("info")
INPUT_DATA_PATH = CURRENT_PATH / config["paths"]["input_data"]

# Paths to the model checkpoints and logs
BEST_SOLUTION_PATH = CURRENT_PATH / config["paths"]["best_solution"]
TEST_SOLUTION_PATH = CURRENT_PATH / config["paths"]["test_solution"]
MODEL_SAVE_PATH = CURRENT_PATH / config["paths"]["model_ckpt"]
LOG_PATH = CURRENT_PATH / config["paths"]["log_dir"]
LP_LOG_PATH = CURRENT_PATH / config["paths"]["lp_log"]
MEMORY_DNN_LOG_PATH = CURRENT_PATH / config["paths"]["memory_dnn_log"]

# Log settings
PULP_LOG_ENABLE = config["logs"]["pulp_log_enable"]
MEMORY_DNN_LOG_ENABLE = config["logs"]["memory_dnn_log_enable"]

# Parameters for the LP model
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

# thread settings
THREADS = config["general"]["threads"]
