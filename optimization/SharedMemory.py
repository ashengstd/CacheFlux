from multiprocessing import shared_memory

import numpy as np

from utils.logger import logger


class SharedMemoryManager:
    def __init__(self):
        self.shared_blocks = {}

    def create_block(self, name: str, array: np.ndarray):
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_array: np.ndarray = np.ndarray(
            array.shape, dtype=array.dtype, buffer=shm.buf
        )
        np.copyto(shared_array, array)
        self.shared_blocks[name] = {
            "shm": shm,
            "shape": array.shape,
            "dtype": array.dtype,
        }
        return shm.name

    def update(self, name: str, array: np.ndarray):
        """更新共享内存"""
        if name not in self.shared_blocks:
            raise ValueError(f"Shared memory block with name {name} does not exist.")
        meta = self.shared_blocks[name]
        shm = meta["shm"]
        shared_array: np.ndarray = np.ndarray(
            meta["shape"], dtype=meta["dtype"], buffer=shm.buf
        )
        np.copyto(shared_array, array)

    def get(self, name: str):
        """获取共享内存中的数组"""
        if name not in self.shared_blocks:
            raise ValueError(f"Shared memory block with name {name} does not exist.")
        meta = self.shared_blocks[name]
        shm = meta["shm"]
        return np.ndarray(meta["shape"], dtype=meta["dtype"], buffer=shm.buf), shm

    def cleanup(self):
        """销毁所有共享内存块"""
        for meta in self.shared_blocks.values():
            shm = meta["shm"]
            shm.close()
            shm.unlink()
        self.shared_blocks.clear()

    def delete_block(self, name: str):
        """删除指定的共享内存块"""
        if name in self.shared_blocks:
            shm = self.shared_blocks[name]["shm"]
            shm.close()
            shm.unlink()
            del self.shared_blocks[name]
        else:
            raise ValueError(f"Shared memory block with name {name} does not exist.")

    def info(self):
        """打印所有共享内存块的信息"""
        for name, meta in self.shared_blocks.items():
            logger.debug(
                f"Name: {name}, Shape: {meta['shape']}, Dtype: {meta['dtype']}, Memory Name: {meta['shm'].name}"
            )


SharedMemManager = SharedMemoryManager()
