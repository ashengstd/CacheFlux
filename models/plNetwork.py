import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class MemoryConfig:
    """Configuration for memory DNN model"""

    network_architecture: List[int]
    learning_rate: float = 0.01
    training_interval: int = 10
    batch_size: int = 100
    memory_size: int = 1000
    beta1: float = 0.09
    beta2: float = 0.999
    weight_decay: float = 0.0001
    max_users: int = 2000


class MemoryBuffer:
    """Memory buffer using lists for flexible user storage"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[
            Tuple[torch.Tensor, torch.Tensor]
        ] = []  # storage (users, caches) the state value is the connectivity of the caches and the action value is the bool of cache whether to be used
        self.ptr = 0

    def add(self, state: torch.Tensor, action: torch.Tensor):
        """Add state-action pair to memory"""
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action))
        else:
            self.buffer[self.ptr] = (state, action)  # 进行循环覆盖

        self.ptr = (self.ptr + 1) % self.max_size  # 更新指针

    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a batch of state-action pairs"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


class MemoryDataset(Dataset):
    """Dataset for memory samples using tensors"""

    def __init__(self, states: torch.Tensor, actions: torch.Tensor):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {"input": self.states[idx], "target": self.actions[idx]}


class plMemoryDNN(pl.LightningModule):
    """Memory DNN implemented using PyTorch Lightning with tensor operations"""

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self._build_network()

        # Initialize memory buffer
        self.register_buffer("memory_counter", torch.tensor(0))
        self.memory_counter: torch.Tensor
        self.memory: MemoryBuffer = MemoryBuffer(
            max_size=self.config.memory_size,
        )

    def _build_network(self):
        """Build neural network architecture"""
        layers = []
        for i in range(len(self.config.network_architecture) - 1):
            layers.extend(
                [
                    nn.Linear(
                        self.config.network_architecture[i],
                        self.config.network_architecture[i + 1],
                    ),
                    nn.ReLU()
                    if i < len(self.config.network_architecture) - 2
                    else nn.Sigmoid(),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)

    def store_memory(
        self,
        state_vector: Union[torch.Tensor, np.ndarray],
        action_vector: Union[torch.Tensor, np.ndarray],
    ):
        """Store state-action pair in memory"""
        # Convert numpy arrays to tensors if necessary
        if isinstance(state_vector, np.ndarray):
            state_vector = torch.from_numpy(state_vector).float()
        if isinstance(action_vector, np.ndarray):
            action_vector = torch.from_numpy(action_vector).float()

        # Move tensors to correct device
        state_vector = state_vector.to(self.device)
        action_vector = action_vector.to(self.device)

        self.memory.add(state_vector, action_vector)
        self.memory_counter = self.memory_counter + torch.tensor(1, device=self.device)

    def encode(
        self,
        state_vector: Union[torch.Tensor, np.ndarray],
        action_vector: Union[torch.Tensor, np.ndarray],
    ):
        """Record training data and trigger training if needed"""
        self.store_memory(state_vector, action_vector)
        if self.memory_counter % self.config.training_interval == 0:
            self.train_memory_batch()

    def train_memory_batch(self):
        """Train on a batch of memory samples"""
        if self.memory.size == 0:
            return

        batch_size = min(self.config.batch_size, self.memory.size)
        states, actions = self.memory.sample(batch_size)

        dataset = MemoryDataset(states, actions)
        dataloader = DataLoader(dataset, batch_size=len(dataset))

        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(self, dataloader)

    def training_step(self, batch):
        """Lightning training step"""
        predictions = self(batch["input"])
        loss = nn.BCELoss()(predictions, batch["target"])
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

    @torch.no_grad()
    def decode(
        self,
        state_vector: Union[torch.Tensor, np.ndarray],
        k: int = 1,
    ) -> torch.Tensor:
        """Generate predictions based on input state"""
        self.eval()

        # Convert to tensor if necessary
        if isinstance(state_vector, np.ndarray):
            state_vector = torch.from_numpy(state_vector).float()

        # Add batch dimension if needed
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)

        state_vector = state_vector.to(self.device)
        predictions = self(state_vector)

        return self.ordinal_policy(predictions, k)

    def ordinal_policy(self, predictions: torch.Tensor, k: int) -> torch.Tensor:
        """Generate binary offloading strategies using ordinal policy"""
        # Initial strategy based on threshold
        strategies = [(predictions > 0.5).float()]

        if k > 1:
            # Calculate margin distances
            margin_distances = torch.abs(predictions - 0.5)

            # Get indices of k-1 closest values to threshold
            _, closest_indices = torch.topk(margin_distances, k - 1, largest=False)

            # Generate additional strategies
            for idx in closest_indices.T:  # Transpose to iterate over columns
                threshold = predictions.gather(1, idx.unsqueeze(1))
                mask = predictions > threshold
                strategies.append(mask.float())

        return torch.stack(strategies, dim=1)

    def plot_training_history(self):
        """Plot training loss history"""
        if not self.trainer.callback_metrics:
            print("No training history available")
            return

        losses = [x["train_loss"].item() for x in self.trainer.callback_metrics]
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(
                0,
                len(losses) * self.config.training_interval,
                self.config.training_interval,
            ),
            losses,
        )
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.grid(True)
        plt.show()

    def save_model(self, save_path: Path):
        """Save model weights and config using pathlib"""
        save_path.mkdir(parents=True, exist_ok=True)  # 创建目录

        # 保存模型权重
        model_path = save_path / "model.pth"
        torch.save(self.state_dict(), model_path)

        # 保存模型配置
        config_path = save_path / "config.json"
        config_path.write_text(json.dumps(asdict(self.config), indent=4))

        print(f"Model saved to {save_path}")

    @staticmethod
    def load_model(load_path: Path, device: torch.device = torch.device("cuda")):
        """Load model weights and config using pathlib"""

        # 加载配置
        config_path = load_path / "config.json"
        config_data = json.loads(config_path.read_text())
        config = MemoryConfig(**config_data)

        # 创建模型实例
        model = plMemoryDNN(config)
        model.to(device)

        # 加载权重
        model_path = load_path / "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))

        print(f"Model loaded from {load_path}")
        return model
