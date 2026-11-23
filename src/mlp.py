"""CPU training for wildfire size regression."""

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    batch_size: int = 256
    loss: str = "mse"
    huber_delta: float = 1.0
    dropout: float = 0.0
    weight_decay: float = 0.0
    batch_norm: bool = False
    max_epochs: int = 80
    patience: int = 10


def build_mlp(input_size, config):
    """Build a network with a single, unrestricted log-size output."""
    layers = []
    for width in config.hidden_sizes:
        layers.append(nn.Linear(input_size, width))
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        input_size = width
    return nn.Sequential(*layers, nn.Linear(input_size, 1), nn.Flatten(0, 1))


def fit_mlp(x_train, y_train, x_val, y_val, config, seed):
    """Fit with AdamW (decoupled weight decay) and restore the epoch with the lowest validation RMSE."""
    torch.manual_seed(seed)
    model = build_mlp(x_train.shape[1], config)
    loss_fn = {"mse": nn.MSELoss(), "huber": nn.HuberLoss(delta=config.huber_delta)}[
        config.loss
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    batches = DataLoader(
        train, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        # Batch normalization requires at least two training examples per batch.
        drop_last=config.batch_norm and len(train) % config.batch_size == 1,
    )
    x_val = torch.as_tensor(x_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.float32)
    best_rmse = float("inf")
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for x, y in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            seen += len(y)

        model.eval()
        with torch.inference_mode():
            residuals = model(x_val) - y_val
            val_rmse = residuals.square().mean().sqrt().item()
        if not math.isfinite(val_rmse):
            raise FloatingPointError(f"Non-finite validation RMSE at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "val_rmse": val_rmse})
        if val_rmse < best_rmse:
            best_epoch, best_rmse = epoch, val_rmse
            best_weights = deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_weights)
    return model, best_epoch, history


@torch.inference_mode()
def predict_log(model, inputs):
    model.eval()
    return model(torch.as_tensor(inputs, dtype=torch.float32)).numpy()
