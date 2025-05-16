import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)  # 输出均值 μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 输出 log(σ²)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)  # μ, log(σ²)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)  # 噪声 ε ~ N(0, 1)
        return mu + eps * std  # z = μ + σ⊙ε
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))  # 输出 [0,1] 之间的像素值
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))  # 展平输入
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar  # 返回重建数据、均值、log(σ²)