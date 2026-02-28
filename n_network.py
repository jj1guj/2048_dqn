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
    def __init__(self, blocks=5, channels=16):
        super().__init__()
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        MOVE_LABEL_NUM = 4
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(MOVE_LABEL_NUM)
        self.linear1 = nn.Linear(4 * 4 * MOVE_LABEL_NUM, MOVE_LABEL_NUM)

    def forward(self, x):
        # (batch, H, W, C) -> (batch, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 16:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3 and x.shape[-1] == 16:
            x = x.permute(2, 0, 1).unsqueeze(0)
        out = self.blocks(x)
        out = F.relu(self.norm1(self.conv1(out)))
        out = out.flatten(1)
        out = self.linear1(out)
        return out
