from torch import nn
from torch.nn import functional as F


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + x)


class N_Network(nn.Module):
    def __init__(self, blocks=3, channels=128):
        super().__init__()

        # 4x4 ボードをそのまま処理（アップサンプリング不要）
        self.input_conv = nn.Sequential(
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            *[ResNetBlock(channels) for _ in range(blocks)]
        )

        # Dueling DQN: Value + Advantage ストリーム
        self.value_stream = nn.Sequential(
            nn.Linear(channels * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(channels * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        # (batch, H, W, C) -> (batch, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 16:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3 and x.shape[-1] == 16:
            x = x.permute(2, 0, 1).unsqueeze(0)

        out = self.input_conv(x)
        out = self.blocks(out)
        out = out.flatten(1)

        value = self.value_stream(out)
        advantage = self.advantage_stream(out)
        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
