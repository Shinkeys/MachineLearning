import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    """
    Simple MLP for approximating sin(x):
    - Input: 1 neuron
    - Hidden layer 1: 16 neurons with ReLU
    - Hidden layer 2: 16 neurons with ReLU
    - Output: 1 neuron (linear)
    """

    def __init__(self):
        super(MLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(1, 16),     # Input layer: 1 -> 16
            nn.ReLU(),            # Activation
            nn.Linear(16, 16),    # Hidden layer: 16 -> 16
            nn.ReLU(),            # Activation
            nn.Linear(16, 16),    # Hidden layer: 16 -> 16
            nn.ReLU(),            # Activation
            nn.Linear(16, 1)      # Output layer: 16 -> 1
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.network(x)


def generate_data(n_train=2048, n_val=256, x_min=-np.pi, x_max=np.pi):
    """Generate training and validation data as PyTorch tensors"""
    x_train = torch.FloatTensor(np.random.uniform(x_min, x_max, n_train).reshape(-1, 1))
    x_val = torch.FloatTensor(np.random.uniform(x_min, x_max, n_val).reshape(-1, 1))
    y_train = torch.FloatTensor(np.sin(x_train.numpy()))
    y_val = torch.FloatTensor(np.sin(x_val.numpy()))
    return (x_train, y_train), (x_val, y_val)


def visualize_realtime_training(train_data, val_data, epochs=1000, learning_rate=1e-2, batch_size=64):
    x_train, y_train = train_data
    x_val, y_val = val_data

    mlp = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    fig, (ax, ax_loss) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    x_smooth = torch.FloatTensor(np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1))
    x_smooth_np = x_smooth.numpy().ravel()
    y_true_smooth = np.sin(x_smooth_np)

    mlp.eval()
    with torch.no_grad():
        y_init = mlp(x_smooth).numpy().ravel()
    
    line_true, = ax.plot(x_smooth_np, y_true_smooth, 'b-', linewidth=2, label='True sin(x)', alpha=0.8)
    line_pred, = ax.plot(x_smooth_np, y_init, 'r--', linewidth=2, label='MLP Prediction')

    scatter_train = ax.scatter(x_train.numpy(), y_train.numpy(), c='cyan', alpha=0.3, s=5, label=f'Train ({len(x_train)} pts)')
    scatter_val = ax.scatter(x_val.numpy(), y_val.numpy(), c='orange', alpha=0.5, s=10, label=f'Val ({len(x_val)} pts)')

    loss_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('x')
    ax.set_ylabel('sin(x)')
    ax.set_title('Real-time MLP Sin(x) Approximation Training (PyTorch)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.2, 1.2)

    ax_loss.set_xlabel('Training Step')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.set_title('Training and Validation Loss Over Time')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale('log')
    
    loss_history = []
    val_loss_history = []
    steps = []
    
    line_train_loss, = ax_loss.plot([], [], 'b-', linewidth=2, label='Training Loss')
    line_val_loss, = ax_loss.plot([], [], 'r-', linewidth=2, label='Validation Loss')
    ax_loss.legend()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=100, min_lr=1e-5)

    def animate(frame):
        mlp.train()
        
        # Mini-batch training
        N = x_train.shape[0]
        if N <= batch_size:
            xb, yb = x_train, y_train
        else:
            idx = torch.randperm(N)[:batch_size]
            xb, yb = x_train[idx], y_train[idx]

        # Forward pass
        y_pred = mlp(xb)
        loss = criterion(y_pred, yb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()

        # Update prediction line
        mlp.eval()
        with torch.no_grad():
            y_pred_smooth = mlp(x_smooth).numpy().ravel()
            y_val_pred = mlp(x_val)
            
            mse_smooth = criterion(torch.FloatTensor(y_pred_smooth.reshape(-1, 1)), 
                                  torch.FloatTensor(y_true_smooth.reshape(-1, 1))).item()
            mse_val = criterion(y_val_pred, y_val).item()
        
        # Update the prediction line (this ensures curves, not dots)
        line_pred.set_ydata(y_pred_smooth)

        # Track loss history
        steps.append(frame)
        loss_history.append(mse_smooth)
        val_loss_history.append(mse_val)

        # Update loss plots
        line_train_loss.set_data(steps, loss_history)
        line_val_loss.set_data(steps, val_loss_history)
        
        # Auto-scale loss plot
        if len(steps) > 1:
            ax_loss.set_xlim(0, max(frame + 10, 100))
            if len(loss_history) > 0:
                all_losses = loss_history + val_loss_history
                min_loss = min(all_losses)
                max_loss = max(all_losses)
                ax_loss.set_ylim(min_loss * 0.5, max_loss * 2)

        # Update learning rate
        scheduler.step(mse_val)
        current_lr = optimizer.param_groups[0]['lr']

        loss_text.set_text(
            f'Epoch: {frame}\n'
            f'Batch Loss: {batch_loss:.6f}\n'
            f'Smooth MSE: {mse_smooth:.6f}\n'
            f'Val MSE: {mse_val:.6f}\n'
            f'LR: {current_lr:.2e}'
        )
        
        return line_pred, loss_text, line_train_loss, line_val_loss

    anim = FuncAnimation(fig, animate, frames=epochs, interval=20, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_data, val_data = generate_data(n_train=512, n_val=128)
    visualize_realtime_training(train_data, val_data, epochs=1000, learning_rate=1e-2, batch_size=64)