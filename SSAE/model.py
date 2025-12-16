import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """编码器：将输入图像编码为隐变量的均值和方差"""
    def __init__(self, input_channels=3, latent_dim=20):
        super(Encoder, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # 全连接层用于计算均值和方差
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # 编码器前向传播
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 计算均值和方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """解码器：将隐变量解码为图像"""
    def __init__(self, latent_dim=20, output_channels=3):
        super(Decoder, self).__init__()
        
        # 全连接层
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # 转置卷积层（上采样）
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z):
        # 解码器前向传播
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # 使用sigmoid将输出限制在[0,1]
        
        return x


class VAE(nn.Module):
    """变分自编码器：结合编码器和解码器"""
    def __init__(self, input_channels=3, latent_dim=20):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧：从N(mu, var)采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        mu, logvar = self.encoder(x)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
    
    def encode(self, x):
        """仅编码，返回隐变量"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z):
        """仅解码，从隐变量生成图像"""
        return self.decoder(z)

