import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.residual_nn import ResidualNN


class TreeNeuralHybridPipeline:
    def __init__(self, max_depth: int = 3, lr: float = 0.01, epochs: int = 500):
        self.base_tree = DecisionTreeRegressor(max_depth=max_depth)
        self.residual_nn = ResidualNN()
        self.lr = lr
        self.epochs = epochs

    def generate_data(self, n_samples: int = 300):
        """Generates 300 synthetic student data points with explicit non-linearities."""
        np.random.seed(42)
        X = np.random.uniform(0, 12, size=(n_samples, 1))
        y_linear = 50 + 4 * X.squeeze()

        # Non-linearities: diminishing returns (>6 hrs) + extreme cramming penalty (>9 hrs)
        non_linear_effect = np.where(X.squeeze() > 6, -1.5 * (X.squeeze() - 6) ** 2, 0)
        non_linear_effect += np.where(X.squeeze() > 9, -10, 0)

        y = y_linear + non_linear_effect + np.random.normal(0, 2, size=n_samples)
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray):
        # 1. Train Base Model (Decision Tree)
        self.base_tree.fit(X, y)
        base_preds = self.base_tree.predict(X)

        # 2. Compute Residuals (Errors)
        residuals = y - base_preds

        # 3. Convert to Tensors for PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        res_tensor = torch.tensor(residuals, dtype=torch.float32).unsqueeze(1)

        # 4. Train Neural Network on Residuals
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.residual_nn.parameters(), lr=self.lr)

        self.residual_nn.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.residual_nn(X_tensor)
            loss = criterion(outputs, res_tensor)
            loss.backward()
            optimizer.step()

        self.residual_nn.eval()
        print("✓ Hybrid Model Trained Successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        tree_preds = self.base_tree.predict(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            nn_corrections = self.residual_nn(X_tensor).numpy().squeeze()

        return tree_preds + nn_corrections

    def plot_results(self, X: np.ndarray, y: np.ndarray):
        X_plot = np.linspace(0, 12, 500).reshape(-1, 1)
        tree_preds = self.base_tree.predict(X_plot)
        hybrid_preds = self.predict(X_plot)

        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='gray', alpha=0.5, label='Actual Data')
        plt.plot(X_plot, tree_preds, label='Tree Only (Macro-Trend)', color='blue', linestyle='--')
        plt.plot(X_plot, hybrid_preds, label='Hybrid Model (Tree + NN Correction)', color='red', linewidth=2)
        plt.title('Tree + Neural Residual Hybrid Pipeline')
        plt.xlabel('Study Hours')
        plt.ylabel('Exam Score')
        plt.legend()
        plt.grid(True)
        plt.savefig('results_vis.png', dpi=300)  # Crucial for README markdown rendering
        plt.show()


if __name__ == "__main__":
    pipeline = TreeNeuralHybridPipeline()
    X, y = pipeline.generate_data()
    pipeline.fit(X, y)
    pipeline.plot_results(X, y)
