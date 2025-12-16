import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from model import VAE


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE损失函数 = 重构损失 + KL散度损失
    
    Args:
        recon_x: 重构的图像
        x: 原始图像
        mu: 隐变量均值
        logvar: 隐变量对数方差
        beta: KL散度的权重（beta-VAE）
    """
    # 重构损失（二元交叉熵）
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度损失（使隐变量分布接近标准高斯分布）
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD, BCE, KLD


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        recon_batch, mu, logvar = model(data)
        
        # 计算损失
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item()/len(data):.4f}, '
                  f'BCE: {bce.item()/len(data):.4f}, '
                  f'KLD: {kld.item()/len(data):.4f}')
    
    avg_loss = train_loss / len(dataloader.dataset)
    avg_bce = train_bce / len(dataloader.dataset)
    avg_kld = train_kld / len(dataloader.dataset)
    
    return avg_loss, avg_bce, avg_kld


def main():
    # 超参数设置
    batch_size = 64
    epochs = 50
    latent_dim = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'使用设备: {device}')
    print(f'隐变量维度: {latent_dim}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小
        transforms.ToTensor(),
    ])
    
    # 加载数据集（这里使用CelebA作为示例，您可以根据需要替换）
    # 如果您有自己的数据集，可以使用ImageFolder
    # dataset = datasets.ImageFolder('path/to/your/dataset', transform=transform)
    
    # 示例：使用CelebA数据集（需要先下载）
    # 如果没有CelebA，可以使用其他数据集或自定义数据集
    try:
        dataset = datasets.CelebA(
            root='./data',
            split='train',
            download=True,
            transform=transform
        )
    except:
        print("无法加载CelebA数据集，请使用自定义数据集")
        print("请将您的人脸数据集放在 ./data/faces 目录下，使用以下代码：")
        print("dataset = datasets.ImageFolder('data/faces', transform=transform)")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 创建模型
    model = VAE(input_channels=3, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        avg_loss, avg_bce, avg_kld = train_epoch(model, dataloader, optimizer, device, epoch)
        
        print(f'Epoch {epoch}/{epochs}, '
              f'平均损失: {avg_loss:.4f}, '
              f'平均BCE: {avg_bce:.4f}, '
              f'平均KLD: {avg_kld:.4f}')
        
        # 每个epoch保存一次模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/vae_epoch_{epoch}.pth')
        
        # 保存一些重构样本
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample = next(iter(dataloader))[0][:8].to(device)
                recon_sample, _, _ = model(sample)
                comparison = torch.cat([sample, recon_sample], dim=0)
                save_image(comparison, f'results/reconstruction_epoch_{epoch}.png', nrow=8)
    
    print("训练完成！")
    print(f"最终模型保存在: checkpoints/vae_epoch_{epochs}.pth")


if __name__ == '__main__':
    import torch.nn.functional as F
    main()

