import json
import os
import tempfile
from multiprocessing import shared_memory
from typing import Dict, Literal

import numpy as np
import pandas as pd

from utils.logger import logger


class SharedMemoryManager:
    def __init__(self):
        """Initializes the SharedMemoryManager."""
        self.shared_blocks = {}

    def _get_default_meta_path(self) -> str:
        """Returns the unified path for the temporary metadata file."""
        return os.path.join(tempfile.gettempdir(), "shared_mem_meta.json")

    def export_metadata(self) -> None:
        """Saves the current shared memory metadata to a system temporary file."""
        meta_dict = {}
        for name, meta in self.shared_blocks.items():
            meta_copy = meta.copy()

            # Check if 'dtype' exists before attempting conversion
            if "dtype" in meta:
                if isinstance(meta["dtype"], np.dtype):
                    meta_copy["dtype"] = str(meta["dtype"])  # Convert dtype to string

            meta_copy["shm"] = meta[
                "shm"
            ].name  # Replace object with memory name string
            meta_dict[name] = meta_copy

        path = self._get_default_meta_path()
        with open(path, "w") as f:
            json.dump(meta_dict, f, indent=4)

        logger.debug(f"Shared memory metadata saved to {path} as a parent process")

    def load_metadata(self) -> None:
        """Loads the shared memory metadata from a system temporary file (used by child processes)."""
        path = self._get_default_meta_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        with open(path, "r") as f:
            meta_dict = json.load(f)

        self.shared_blocks.clear()
        for name, meta in meta_dict.items():
            shm = shared_memory.SharedMemory(
                name=meta["shm"]
            )  # Restore shared memory by name
            meta["shm"] = shm
            self.shared_blocks[name] = meta

        logger.debug(f"Shared memory metadata loaded from {path} as a child process")

    def create_block(self, name: str, data: np.ndarray | pd.DataFrame | Dict) -> str:
        if isinstance(data, np.ndarray):
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shared_data: np.ndarray = np.ndarray(
                data.shape, dtype=data.dtype, buffer=shm.buf
            )
            np.copyto(shared_data, data)
            self.shared_blocks[name] = {
                "shm": shm,
                "shape": data.shape,
                "dtype": data.dtype,
                "type": "ndarray",
            }
            return shm.name

        elif isinstance(data, pd.DataFrame):
            shm = shared_memory.SharedMemory(create=True, size=data.values.nbytes)
            shared_array: np.ndarray = np.ndarray(
                data.shape, dtype=data.values.dtype, buffer=shm.buf
            )
            np.copyto(shared_array, data.values)
            self.shared_blocks[name] = {
                "shm": shm,
                "shape": data.shape,
                "dtype": data.values.dtype,
                "type": "dataframe",
                "index": data.index.tolist(),
                "columns": data.columns.tolist(),
            }
            return shm.name

        elif isinstance(data, dict):
            json_str = json.dumps(data)
            json_bytes = json_str.encode("utf-8")
            shm = shared_memory.SharedMemory(create=True, size=len(json_bytes))
            buffer = shm.buf
            buffer[: len(json_bytes)] = json_bytes
            self.shared_blocks[name] = {
                "shm": shm,
                "size": len(json_bytes),
                "type": "json",
            }
            return shm.name
        else:
            raise TypeError("Data must be a NumPy ndarray or a pandas DataFrame.")

    def update_block(self, name: str, data: np.ndarray | pd.DataFrame) -> None:
        if name not in self.shared_blocks.keys():
            raise ValueError(f"Shared memory block with name {name} does not exist.")

        meta = self.shared_blocks[name]
        shm = meta["shm"]
        shared_array: np.ndarray
        if isinstance(data, np.ndarray):
            shared_array = np.ndarray(
                meta["shape"], dtype=meta["dtype"], buffer=shm.buf
            )
            np.copyto(shared_array, data)

        elif isinstance(data, pd.DataFrame):
            shared_array = np.ndarray(
                meta["shape"], dtype=meta["dtype"], buffer=shm.buf
            )
            np.copyto(shared_array, data.values)

        elif isinstance(data, dict):
            json_str = json.dumps(data)
            json_bytes = json_str.encode("utf-8")
            if len(json_bytes) > meta["size"]:
                raise ValueError("New JSON data exceeds original shared memory size.")
            shm.buf[: len(json_bytes)] = json_bytes
            meta["size"] = len(json_bytes)  # update stored size

        else:
            raise TypeError("Data must be a NumPy ndarray or a pandas DataFrame.")

    def get_by_name(self, name: str) -> np.ndarray | pd.DataFrame | Dict | None:
        if name not in self.shared_blocks:
            raise ValueError(f"Shared memory block with name {name} does not exist.")

        meta = self.shared_blocks[name]
        shm = meta["shm"]

        if meta["type"] == "ndarray":
            return np.ndarray(meta["shape"], dtype=meta["dtype"], buffer=shm.buf)

        elif meta["type"] == "dataframe":
            index = pd.Index(meta["index"])
            columns = pd.Index(meta["columns"])
            array: np.ndarray = np.ndarray(
                meta["shape"], dtype=meta["dtype"], buffer=shm.buf
            )
            return pd.DataFrame(array, index=index, columns=columns)

        elif meta["type"] == "json":
            json_bytes = bytes(shm.buf[: meta["size"]])
            json_obj = json.loads(json_bytes.decode("utf-8"))
            return json_obj

        else:
            return None

    def cleanup(self) -> None:
        for meta in self.shared_blocks.values():
            shm = meta["shm"]
            shm.close()
            shm.unlink()
        self.shared_blocks.clear()

    def delete_block(self, name: str) -> None:
        if name in self.shared_blocks:
            shm = self.shared_blocks[name]["shm"]
            shm.close()
            shm.unlink()
            del self.shared_blocks[name]
        else:
            raise ValueError(f"Shared memory block with name {name} does not exist.")

    def info(self, type: Literal["print", "logger"] = "logger") -> None:
        if type == "print":
            for name, meta in self.shared_blocks.items():
                if meta["type"] == "json":
                    print(f"Name: {name}, Size: {meta['size']} bytes")
                else:
                    print(
                        f"Name: {name}, Shape: {meta['shape']}, Dtype: {meta['dtype']}, Memory Name: {meta['shm'].name}"
                    )
        elif type == "logger":
            for name, meta in self.shared_blocks.items():
                if meta["type"] == "json":
                    logger.debug(f"Name: {name}, Size: {meta['size']} bytes")
                else:
                    logger.debug(
                        f"Name: {name}, Shape: {meta['shape']}, Dtype: {meta['dtype']}, Memory Name: {meta['shm'].name}"
                    )


SharedMemManager = SharedMemoryManager()
