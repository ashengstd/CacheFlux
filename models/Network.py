import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        network_architecture,
        learning_rate=0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        log_file=None,
        log_enable=False,
    ):
        self.net = network_architecture
        self.learning_rate = learning_rate
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.log_file = log_file
        self.log_enable = log_enable

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize memory attributes
        self.memory_counter = 0
        self.memory = []
        self.cost_history = []

        # Build neural network
        self._build_net()

    def _build_net(self):
        """Build the neural network architecture."""
        self.model = nn.Sequential(
            nn.Linear(self.net[0], self.net[1]),
            nn.ReLU(),
            nn.Linear(self.net[1], self.net[2]),
            nn.ReLU(),
            nn.Linear(self.net[2], self.net[3]),
            nn.Sigmoid(),
        ).to(self.device)

    def remember(self, h, m):
        """Store the input-output pair in memory."""
        idx = self.memory_counter % self.memory_size
        combined_matrix = np.hstack((h, m))
        if len(self.memory) < self.memory_size:
            self.memory.append(combined_matrix)
        else:
            self.memory[idx] = combined_matrix
        self.memory_counter += 1

    def encode(self, h, m):
        """Record the current training data and update the model periodically."""
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        """Update model parameters based on sampled memory."""
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size, replace=False
            )
        else:
            num_samples = min(self.memory_counter, self.batch_size)
            sample_index = np.random.choice(
                self.memory_counter, size=num_samples, replace=False
            )

        batch_memory = [self.memory[i] for i in sample_index]
        combined_matrix = np.vstack(batch_memory)
        h_len = self.net[0]
        h_train = torch.Tensor(combined_matrix[:, :h_len]).to(self.device)
        m_train = torch.Tensor(combined_matrix[:, h_len:]).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.09, 0.999),
            weight_decay=0.0001,
        )
        criterion = nn.BCELoss().to(self.device)

        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        self.cost_history.append(self.cost)

        if self.log_enable:
            with open(self.log_file, "a") as f:
                f.write(f"Loss: {self.cost}\n")

        assert self.cost > 0, "Cost should be greater than 0"

    def decode(self, h, k=1, mode="OP"):
        """Generate predictions based on the input and mode."""
        h = torch.Tensor(h[np.newaxis, :]).to(self.device)
        self.model.eval()
        m_pred = self.model(h).cpu().detach().numpy()

        if mode == "OP":
            return self.knm(m_pred, k)
        elif mode == "KNN":
            return self.knn(m_pred, k)
        else:
            raise ValueError("The action selection must be 'OP' or 'KNN'")

    def knm(self, m1, k=1):
        """Generate binary offloading strategies using the K*K ordinal policy."""
        m1 = m1.squeeze()
        m_list = []

        for m in m1:
            sample_list = [1 * (m > 0.5)]
            if k > 1:
                m_abs = abs(m - 0.5)
                idx_list = np.argsort(m_abs)[: k - 1]
                for i in range(k - 1):
                    if m[idx_list[i]] > 0.5:
                        sample_list.append(1 * (m - m[idx_list[i]] > 0))
                    else:
                        sample_list.append(1 * (m - m[idx_list[i]] >= 0))
            m_list.append(sample_list)

        return m_list

    def plot_cost(self):
        """Plot the training loss over time."""
        plt.plot(
            np.arange(len(self.cost_history)) * self.training_interval,
            self.cost_history,
        )
        plt.ylabel("Training Loss")
        plt.xlabel("Time Frames")
        plt.title("Training Loss Over Time")
        plt.show()
