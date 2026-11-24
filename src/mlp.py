"""TensorFlow/Keras training for wildfire size regression."""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras


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
    layers = [keras.Input(shape=(input_size,))]
    for width in config.hidden_sizes:
        layers.append(keras.layers.Dense(width))
        if config.batch_norm:
            layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.ReLU())
        if config.dropout:
            layers.append(keras.layers.Dropout(config.dropout))
    layers.append(keras.layers.Dense(1))
    return keras.Sequential(layers)


def fit_mlp(x_train, y_train, x_val, y_val, config, seed):
    """Fit with AdamW and restore the weights with the lowest validation RMSE."""
    keras.utils.set_random_seed(seed)
    losses = {
        "mse": keras.losses.MeanSquaredError(),
        "huber": keras.losses.Huber(delta=config.huber_delta),
    }
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_rmse", mode="min", patience=config.patience,
        restore_best_weights=True,
    )
    # Match the scalar output with a column of targets for losses and metrics.
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
    with tf.device("/CPU:0"):
        model = build_mlp(x_train.shape[1], config)
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            ),
            loss=losses[config.loss],
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        metrics = model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            batch_size=config.batch_size, epochs=config.max_epochs, shuffle=True,
            callbacks=[keras.callbacks.TerminateOnNaN(), early_stopping], verbose=0,
        ).history

    history = [
        {"epoch": epoch, "train_loss": loss, "val_rmse": rmse}
        for epoch, (loss, rmse) in enumerate(zip(metrics["loss"], metrics["val_rmse"]), start=1)
    ]
    for epoch in history:
        if not np.isfinite([epoch["train_loss"], epoch["val_rmse"]]).all():
            raise FloatingPointError(
                f"Non-finite training loss or validation RMSE at epoch {epoch['epoch']}"
            )
    best_epoch = min(history, key=lambda epoch: epoch["val_rmse"])["epoch"]
    return model, best_epoch, history


def predict_log(model, inputs):
    """Return one log-size prediction per row."""
    with tf.device("/CPU:0"):
        return model(inputs, training=False).numpy().ravel()
