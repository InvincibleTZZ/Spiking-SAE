import sys
import os

if 'model_spike_vae' in sys.modules:
    del sys.modules['model_spike_vae']

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
from datetime import datetime

from model_spike_vae import SpikeVAE, spike_vae_loss, analyze_spike_patterns


class ImageDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
        self.image_paths = []
        
        image_paths_set = set()
        for filename in os.listdir(root_dir):
            file_path = os.path.join(root_dir, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower().lstrip('.')
                if ext in image_extensions:
                    image_paths_set.add(file_path)
        
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
            return image, 0
        except Exception as e:
            print(f"加载图片失败 {img_path}: {e}")
            if self.transform:
                return self.transform(Image.new('RGB', (64, 64), (0, 0, 0))), 0
            return Image.new('RGB', (64, 64), (0, 0, 0)), 0


def main():
    # ========== 配置参数 ==========
    DATA_DIR = 'D:/lyk/VAE/data/faces'
    IMAGE_SIZE = 128
    BATCH_SIZE = 16
    
    LATENT_DIM = 256
    PRIOR_SPARSITY = 0.1
    
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-4
    BETA = 1.0
    SPARSITY_WEIGHT = 0.05
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = 'checkpoints_spike_vae_fixed'
    VAL_SPLIT = 0.1
    # ==============================
    
    print("=" * 60)
    print("✅ Spike-VAE 最终修复版训练程序")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {DEVICE}")
    print(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"隐变量维度: {LATENT_DIM}")
    print(f"期望稀疏度: {PRIOR_SPARSITY:.1%}")
    print("=" * 60)
    
    # 检查数据集
    if not os.path.exists(DATA_DIR):
        print(f'\n错误: 数据集目录不存在: {DATA_DIR}')
        return
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    # 加载数据集
    print(f'\n加载数据集: {DATA_DIR}')
    full_dataset = ImageDataset(DATA_DIR, transform=transform)
    
    # 分割数据集
    if VAL_SPLIT > 0:
        val_size = int(len(full_dataset) * VAL_SPLIT)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        print(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f'训练集: {len(train_dataset)}')
    
    # 创建数据加载器（Windows使用num_workers=0）
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # 改为False，更安全
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
    
    # 创建保存目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('results_spike_vae_fixed', exist_ok=True)
    
    # 创建模型
    print('\n创建模型...')
    device = torch.device(DEVICE)
    model = SpikeVAE(
        input_channels=3,
        latent_dim=LATENT_DIM,
        prior_sparsity=PRIOR_SPARSITY,
        image_size=IMAGE_SIZE
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 保存测试样本
    test_sample = None
    for data, _ in train_loader:
        test_sample = data[:8].to(device)
        break
    
    print('\n开始训练...')
    print('=' * 60)
    
    # 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        train_actual_sparse = 0
        batch_count = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # 确保数据有效
            data = torch.clamp(data, 0.0, 1.0)
            if torch.isnan(data).any():
                continue
            
            try:
                # 前向传播
                recon_batch, firing_probs, spikes = model(data, deterministic=False)
                
                # !!! 关键检查：在损失计算前再次确认firing_probs范围
                if (firing_probs < 0).any() or (firing_probs > 1).any():
                    print(f'\n⚠️  检测到firing_probs超出范围!')
                    print(f'   范围: [{firing_probs.min().item()}, {firing_probs.max().item()}]')
                    print(f'   强制裁剪并继续...')
                    firing_probs = torch.clamp(firing_probs, 0.0, 1.0)
                
                # 计算损失
                loss, recon_loss, kl_loss, sparsity_loss, actual_sparsity = spike_vae_loss(
                    recon_batch, data, firing_probs, spikes,
                    prior_sparsity=model.prior_sparsity,
                    beta=BETA,
                    sparsity_weight=SPARSITY_WEIGHT
                )
                
                # 检查损失有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f'\n⚠️  损失无效，跳过batch {batch_idx}')
                    continue
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 累计
                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
                train_actual_sparse += actual_sparsity.item()
                batch_count += 1
                
                # 日志
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch}/{NUM_EPOCHS} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)] '
                          f'Loss: {loss.item():.4f} | '
                          f'Recon: {recon_loss.item():.4f} | '
                          f'KL: {kl_loss.item():.4f} | '
                          f'Sparsity: {actual_sparsity.item():.2%}')
            
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'assert' in str(e).lower():
                    print(f'\n❌ CUDA错误在epoch {epoch}, batch {batch_idx}')
                    print(f'   错误: {e}')
                    print(f'   跳过此batch并继续...')
                    continue
                else:
                    raise e
        
        # 计算平均
        if batch_count > 0:
            avg_train_loss = train_loss / batch_count
            avg_train_recon = train_recon / batch_count
            avg_train_kl = train_kl / batch_count
            avg_train_actual_sparse = train_actual_sparse / batch_count
        else:
            print(f'\n⚠️  Epoch {epoch}: 所有batch都失败了！')
            continue
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_actual_sparse = 0
            val_count = 0
            
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    data = torch.clamp(data, 0.0, 1.0)
                    
                    try:
                        recon_batch, firing_probs, spikes = model(data, deterministic=True)
                        loss, _, _, _, actual_sparsity = spike_vae_loss(
                            recon_batch, data, firing_probs, spikes,
                            prior_sparsity=model.prior_sparsity,
                            beta=BETA,
                            sparsity_weight=SPARSITY_WEIGHT
                        )
                        
                        val_loss += loss.item()
                        val_actual_sparse += actual_sparsity.item()
                        val_count += 1
                    except:
                        continue
            
            if val_count > 0:
                avg_val_loss = val_loss / val_count
                avg_val_sparse = val_actual_sparse / val_count
                scheduler.step(avg_val_loss)
                
                print(f'\n{"="*60}')
                print(f'Epoch {epoch} Summary:')
                print(f'  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
                print(f'  Train Sparsity: {avg_train_actual_sparse:.2%} | Val Sparsity: {avg_val_sparse:.2%}')
                print(f'  Target Sparsity: {PRIOR_SPARSITY:.2%}')
                print(f'{"="*60}\n')
        else:
            print(f'\n{"="*60}')
            print(f'Epoch {epoch}: Loss: {avg_train_loss:.4f}, Sparsity: {avg_train_actual_sparse:.2%}')
            print(f'{"="*60}\n')
        
        # 保存重构样本
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                try:
                    recon_sample, firing_probs_sample, spikes_sample = model(test_sample, deterministic=True)
                    
                    # 简单对比图
                    comparison = torch.cat([test_sample, recon_sample], dim=0)
                    save_image(comparison, 
                              f'results_spike_vae_fixed/reconstruction_epoch_{epoch}.png', 
                              nrow=8)
                    
                    recon_error = F.mse_loss(recon_sample, test_sample).item()
                    actual_sparsity_sample = spikes_sample.mean().item()
                    
                    print(f'\n✓ 已保存重构样本 (Epoch {epoch}):')
                    print(f'  - reconstruction_epoch_{epoch}.png')
                    print(f'  - 重构误差 (MSE): {recon_error:.6f}')
                    print(f'  - 稀疏度: {actual_sparsity_sample:.2%}')
                except Exception as e:
                    print(f'\n⚠️  保存重构样本失败: {e}')
        
        # 保存模型
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'spike_vae_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'latent_dim': LATENT_DIM,
                'prior_sparsity': PRIOR_SPARSITY,
                'beta': BETA,
                'sparsity_weight': SPARSITY_WEIGHT,
                'image_size': IMAGE_SIZE,
            }, checkpoint_path)
            print(f'✓ 模型已保存: {checkpoint_path}')
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"模型保存在: {CHECKPOINT_DIR}/")
    print(f"重构样本在: results_spike_vae_fixed/")


if __name__ == '__main__':
    print("\n🔄 强制重新加载模块...")
    print("   这确保使用最新的修复代码\n")
    main()

