import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """增强版编码器：更大的容量以提升重构质量"""
    def __init__(self, input_channels=3, latent_dim=128, image_size=256):
        super(Encoder, self).__init__()
        
        # 根据输入尺寸决定卷积层数（增加通道数以提升表达能力）
        if image_size == 64:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            self.fc_size = 512 * 4 * 4
            self.num_layers = 4
        elif image_size == 128:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            self.fc_size = 512 * 8 * 8
            self.num_layers = 4
        elif image_size == 256:
            # 256x256: 增加通道数以提升表达能力
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.fc_size = 512 * 8 * 8
            self.num_layers = 5
        elif image_size == 400:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.fc_size = 512 * 12 * 12
            self.num_layers = 6
        else:
            raise ValueError(f"不支持的图像尺寸: {image_size}")
        
        # Batch Normalization 提升训练稳定性
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 全连接层用于计算均值和方差
        self.fc_mu = nn.Linear(self.fc_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_size, latent_dim)
        self.image_size = image_size
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        if self.num_layers >= 5:
            x = F.relu(self.bn4(self.conv5(x)))
        if self.num_layers >= 6:
            x = F.relu(self.bn4(self.conv6(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 计算均值和方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """增强版解码器：更大的容量以提升重构质量"""
    def __init__(self, latent_dim=128, output_channels=3, image_size=256):
        super(Decoder, self).__init__()
        
        # 根据输出尺寸决定转置卷积层数（增加通道数）
        if image_size == 64:
            self.fc_size = 512 * 4 * 4
            self.start_channels = 512
            self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
            self.num_layers = 4
        elif image_size == 128:
            self.fc_size = 512 * 8 * 8
            self.start_channels = 512
            self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
            self.num_layers = 4
        elif image_size == 256:
            self.fc_size = 512 * 8 * 8
            self.start_channels = 512
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.deconv5 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
            self.num_layers = 5
        elif image_size == 400:
            self.fc_size = 512 * 12 * 12
            self.start_channels = 512
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.deconv6 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
            self.num_layers = 6
        else:
            raise ValueError(f"不支持的图像尺寸: {image_size}")
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 全连接层
        self.fc = nn.Linear(latent_dim, self.fc_size)
        self.image_size = image_size
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.start_channels, 
                   int((self.fc_size // self.start_channels) ** 0.5),
                   int((self.fc_size // self.start_channels) ** 0.5))
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        
        if self.num_layers >= 5:
            x = F.relu(self.deconv5(x))
        if self.num_layers >= 6:
            x = F.relu(self.deconv6(x))
        
        x = torch.sigmoid(x)  # 使用sigmoid将输出限制在[0,1]
        
        return x


class VAE(nn.Module):
    """增强版变分自编码器：更大的容量以提升重构质量"""
    def __init__(self, input_channels=3, latent_dim=128, image_size=256):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_channels, latent_dim, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)
        self.latent_dim = latent_dim
        self.image_size = image_size
        
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

