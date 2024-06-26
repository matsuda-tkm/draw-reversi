from torch import nn


class ValueNetwork(nn.Module):
    def __init__(self, hidden_channels: int, conv_layers: int):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(8, hidden_channels, kernel_size=5, padding=2), nn.ReLU()
        )

        hidden_layers = []
        for _ in range(conv_layers):
            hidden_layers.extend(
                [
                    nn.Conv2d(
                        hidden_channels, hidden_channels, kernel_size=3, padding=1
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(hidden_channels),
                ]
            )
        for _ in range(2):
            hidden_layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=1)
            )
        hidden_layers.append(nn.Flatten())
        self.hidden_layer = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels * 144, 128),  # incase of input=(8, 8, 8)
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out.tanh()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualValueNetwork(nn.Module):
    def __init__(self, hidden_channels: int, conv_layers: int):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(8, hidden_channels, kernel_size=5, padding=2), nn.ReLU()
        )

        # Building the hidden layers with residual blocks
        self.hidden_layers = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(conv_layers)]
        )

        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 8 * 8, 128),  # Adjusted for input size (8, 8)
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out.tanh()


if __name__ == "__main__":
    import torch

    model = ResidualValueNetwork(hidden_channels=8, conv_layers=5)
    # model = ValueNetwork(hidden_channels=8, conv_layers=5)
    x = torch.randn(1, 8, 8, 8)
    with torch.no_grad():
        output = model(x)
    print(output)

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f"Total parameters: {total_params}")
