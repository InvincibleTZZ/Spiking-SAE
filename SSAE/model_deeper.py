import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """更深层的编码器：增加深度和通道数以提升表达能力"""
    def __init__(self, input_channels=3, latent_dim=128, image_size=256):
        super(Encoder, self).__init__()
        
        # 对于256x256，使用更深的网络
        # 64x64: 4层 (64->32->16->8->4)
        # 128x128: 4层 (128->64->32->16->8)
        # 256x256: 5层 (256->128->64->32->16->8)
        # 可以增加到6层以获得更好的特征提取
        
        if image_size == 64:
            # 64x64: 4层，但增加通道数
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            self.fc_size = 512 * 4 * 4
            self.num_layers = 4
        elif image_size == 128:
            # 128x128: 5层（增加一层）
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.fc_size = 512 * 4 * 4
            self.num_layers = 5
        elif image_size == 256:
            # 256x256: 6层（增加一层，更深）
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
            self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.fc_size = 512 * 4 * 4  # 256->128->64->32->16->8->4
            self.num_layers = 6
        else:
            raise ValueError(f"不支持的图像尺寸: {image_size}")
        
        # Batch Normalization 提升训练稳定性
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 全连接层
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
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """更深层的解码器：对称的深度网络"""
    def __init__(self, latent_dim=128, output_channels=3, image_size=256):
        super(Decoder, self).__init__()
        
        if image_size == 64:
            self.fc_size = 512 * 4 * 4
            self.start_channels = 512
            self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
            self.num_layers = 4
        elif image_size == 128:
            self.fc_size = 512 * 4 * 4
            self.start_channels = 512
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.deconv5 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
            self.num_layers = 5
        elif image_size == 256:
            self.fc_size = 512 * 4 * 4
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
        
        x = torch.sigmoid(x)
        
        return x


class VAE(nn.Module):
    """更深层的VAE：增加深度以提升特征提取能力"""
    def __init__(self, input_channels=3, latent_dim=128, image_size=256):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_channels, latent_dim, image_size)
        self.decoder = Decoder(latent_dim, input_channels, image_size)
        self.latent_dim = latent_dim
        self.image_size = image_size
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z):
        return self.decoder(z)

