config = {
    "hidden_channels": 8,
    "conv_layers": 7,
    "optimizer": "AdamW",
    "lr": 0.01,
    "scheduler": "ExponentialLR",
    "gamma": 0.97,
    "criterion": "HuberLoss",
    "batch_size": 8192,
    "seed": 42,
    "augmentation": True,
    "n_epoch": 10,
    "n_data": 20,
}
