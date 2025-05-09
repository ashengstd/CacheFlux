import logging

from pytorch_lightning.loggers import Logger
from rich.logging import RichHandler


class RichLogger(Logger):
    def __init__(self, logger):
        super().__init__()
        self._logger = logger

    @property
    def name(self):
        return "rich_logger"

    @property
    def version(self):
        return "v0.1"

    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            self._logger.info(f"Step {step} - {k}: {v}")

    def log_hyperparams(self, params):
        self._logger.info(f"Hyperparameters: {params}")

    def finalize(self, status):
        self._logger.info(f"Training finished with status: {status}")


# 设置 rich 日志处理器
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger("DRLPO")
