import pathlib
import tomllib

ROOT_PATH = pathlib.Path(__file__).parent.parent
config_path = ROOT_PATH / "config.toml"

with open(config_path, "rb") as f:
    config = tomllib.load(f)

# Constants
CACHES = config["constants"]["CACHES"]
N = config["constants"]["N"]
PEEK_PERIOD = config["constants"]["PEEK_PERIOD"]
MAX_SAMPLES_PER_DAY = config["constants"]["MAX_SAMPLES_PER_DAY"]

# Paths to the data files and directories
CSV_PATH = ROOT_PATH / config["paths"]["csv_data"]
REQ_CSV_CSV_PATH = CSV_PATH.joinpath("request")
INFO_CSV_PATH = CSV_PATH.joinpath("info")

# Paths to the preprocessed data and directories
PRE_DATA_PATH = ROOT_PATH / config["paths"]["preprocessed_data"]
INFO_NPY_PATH = PRE_DATA_PATH.joinpath("npy")
INPUT_DATA_PATH = PRE_DATA_PATH.joinpath("mat")

# Paths to the model checkpoints
MODEL_SAVE_PATH = ROOT_PATH / config["paths"]["model_ckpt"]

# Paths to the log files
LOG_PATH = ROOT_PATH / config["paths"]["log_dir"]

# Parameters for the LP model
PL_PARAMS = config["params"]
