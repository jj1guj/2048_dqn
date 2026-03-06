import torch
from torch import nn
from torch.nn import functional as F

class NoisyLinear(nn.Module):
    weight_epsilon: torch.Tensor
    bias_epsilon: torch.Tensor

    def __init__(self, in_features, out_features, std_init=0.5, sigma_min=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sigma_min = sigma_min

        # 平均パラメータ
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # ノイズの標準偏差パラメータ
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # ノイズバッファ（学習しない）
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        bound = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        
        # sigmaは固定値で初期化
        sig = self.std_init / (self.in_features ** 0.5)
        self.weight_sigma.data.fill_(sig)
        self.bias_sigma.data.fill_(sig)
    
    def sample_noise(self):
        """ノイズをサンプリング（バッチごとに呼び出し）"""
        # f(x) = sign(x) * sqrt(|x|)
        def scale_noise(x):
            return torch.sign(x) * torch.sqrt(torch.abs(x))

        device = self.weight_mu.device
        
        # 入出力のノイズを生成
        in_eps = scale_noise(torch.randn((self.in_features,), device=device))
        out_eps = scale_noise(torch.randn((self.out_features,), device=device))
        
        # 外積でノイズ行列を作成
        self.weight_epsilon.copy_(torch.outer(out_eps, in_eps))
        self.bias_epsilon.copy_(out_eps)
    
    def forward(self, x):
        # ノイズがサンプルされていない場合は決定的で実行
        if not self.training:
            return torch.nn.functional.linear(x, self.weight_mu, self.bias_mu)

        # σに下限を設ける
        weight_sigma = self.weight_sigma.clamp(min=self.sigma_min)
        bias_sigma = self.bias_sigma.clamp(min=self.sigma_min)

        # ノイズを含めた重み計算
        weight = self.weight_mu + weight_sigma * self.weight_epsilon
        bias = self.bias_mu + bias_sigma * self.bias_epsilon
        
        return torch.nn.functional.linear(x, weight, bias)

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

        self.conv = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=3, padding=1),  # (128, 4, 4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # (128, 4, 4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # (128, 4, 4)
            nn.ReLU(),
        )
        # 128 * 4 * 4 = 2048
        self.shared = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(512, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 128),
            nn.ReLU(),
            NoisyLinear(128, 4),
        )

    def forward(self, x):
        # (batch, 4, 4, 16) → (batch, 256)
        out = x.reshape(x.size(0) if x.dim() == 4 else 1, -1).float()
        out = self.shared(out)

        value = self.value_stream(out)
        advantage = self.advantage_stream(out)
        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
    
    def reset_noise(self):
        # 全NoisyLinear層のノイズをリサンプル
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.sample_noise()
