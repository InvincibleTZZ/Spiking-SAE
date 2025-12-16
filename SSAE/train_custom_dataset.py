"""
使用自定义人脸数据集训练VAE的示例脚本
请将您的人脸数据集放在 data/faces 目录下
数据集结构应该是：
data/faces/
  ├── image1.jpg
  ├── image2.jpg
  └── ...
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import glob


class ImageDataset(Dataset):
    """自定义数据集类，用于加载单层文件夹中的图片"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 支持的图片格式（不区分大小写）
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
        self.image_paths = []
        
        # 使用set去重，避免重复计算
        image_paths_set = set()
        
        # 收集所有图片文件（只搜索当前目录，不搜索子目录）
        for filename in os.listdir(root_dir):
            file_path = os.path.join(root_dir, filename)
            # 只处理文件，不处理目录
            if os.path.isfile(file_path):
                # 获取文件扩展名（不区分大小写）
                ext = os.path.splitext(filename)[1].lower().lstrip('.')
                if ext in image_extensions:
                    image_paths_set.add(file_path)  # 使用set自动去重
        
        # 转换为列表并排序（保证顺序一致）
        self.image_paths = sorted(list(image_paths_set))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在 {root_dir} 中未找到任何图片文件！")
        
        print(f"找到 {len(self.image_paths)} 张图片")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0  # 返回0作为标签（VAE不需要标签）
        except Exception as e:
            print(f"加载图片失败 {img_path}: {e}")
            # 返回一个黑色图片作为占位符
            if self.transform:
                return self.transform(Image.new('RGB', (256, 256), (0, 0, 0))), 0
            return Image.new('RGB', (256, 256), (0, 0, 0)), 0


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE损失函数"""
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD


def train_epoch(model, dataloader, optimizer, device, epoch, beta=1.0):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, beta=beta)
        
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
    # ========== 配置参数 ==========
    # 图像尺寸：64, 128, 256, 400
    IMAGE_SIZE = 256  # 修改这里改变训练图像尺寸（推荐256，400需要更多显存）
    # ==============================
    
    # 超参数设置
    # 根据图像尺寸调整batch size（高分辨率需要更小的batch size）
    if IMAGE_SIZE <= 64:
        batch_size = 64
    elif IMAGE_SIZE <= 128:
        batch_size = 32
    elif IMAGE_SIZE <= 256:
        batch_size = 16
    else:  # 400
        batch_size = 8  # 400x400需要很小的batch size
    
    epochs = 100  # 增加训练轮数以提升重构质量
    latent_dim = 128  # 增加隐变量维度以提升表达能力（从20增加到128）
    learning_rate = 0.001  # 稍微降低学习率以获得更稳定的训练
    beta = 0.01  # beta-VAE参数：降低beta可以提升重构质量（减少KLD约束）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'使用设备: {device}')
    print(f'隐变量维度: {latent_dim}')
    print(f'图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}')
    print(f'Batch size: {batch_size}')
    print(f'训练轮数: {epochs}')
    print(f'学习率: {learning_rate}')
    print(f'Beta (KLD权重): {beta}')
    print('=' * 60)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 调整图像大小
        transforms.ToTensor(),
    ])
    
    # 使用自定义数据集
    dataset_path = 'D:/lyk/VAE/data/faces'
    if not os.path.exists(dataset_path):
        print(f'错误: 数据集目录不存在: {dataset_path}')
        print('请创建该目录并将您的人脸图片放入其中')
        return
    
    # 使用自定义数据集类加载单层文件夹中的图片
    dataset = ImageDataset(dataset_path, transform=transform)
    print(f'数据集大小: {len(dataset)} 张图片')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 创建模型（根据图像尺寸选择模型）
    if IMAGE_SIZE > 64:
        from model_highres import VAE
        model = VAE(input_channels=3, latent_dim=latent_dim, image_size=IMAGE_SIZE).to(device)
    else:
        from model import VAE
        model = VAE(input_channels=3, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        avg_loss, avg_bce, avg_kld = train_epoch(model, dataloader, optimizer, device, epoch, beta)
        
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
            'latent_dim': latent_dim,
            'image_size': IMAGE_SIZE,  # 保存图像尺寸，供推理时使用
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
    main()

