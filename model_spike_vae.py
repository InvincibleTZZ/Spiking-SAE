"""
Spike-VAE: 全脉冲变分自编码器
适用于脉冲神经网络(SNN)的VAE变体

核心特性:
1. 伯努利隐变量分布（而非高斯分布）
2. 稀疏脉冲编码（兼具SAE稀疏自编码特性）
3. 使用Straight-Through Estimator (STE)实现梯度传播
4. 可调节的稀疏性水平

数学原理:
- 传统VAE: q(z|x) = N(μ, σ²), z ∈ R^d
- Spike-VAE: q(z|x) = Bernoulli(π), z ∈ {0,1}^d
- KL散度: KL[q(z|x) || p(z)] = Σ π*log(π/p) + (1-π)*log((1-π)/(1-p))
- 稀疏性: L_sparse = λ * ||z||_1 (L1惩罚)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticBinarize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for 二值化
    
    前向传播: 随机二值化（根据伯努利分布采样）
    反向传播: 直通梯度（恒等映射）
    
    这允许我们训练离散的脉冲神经网络
    """
    @staticmethod
    def forward(ctx, input):
        """
        前向: 从Bernoulli(input)采样二值输出
        input: 发放概率 π ∈ [0,1]
        output: 二值脉冲 z ∈ {0,1}
        """
        # 关键修复：确保概率严格在[0,1]范围内
        # 防止CUDA assert错误
        input = torch.clamp(input, 0.0, 1.0)
        
        # 随机二值化：z ~ Bernoulli(π)
        output = torch.bernoulli(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向: Straight-Through - 梯度直接传递
        """
        # 直通梯度，不做任何修改
        return grad_output


class DeterministicBinarize(torch.autograd.Function):
    """
    确定性二值化（用于推理）
    阈值: 0.5
    """
    @staticmethod
    def forward(ctx, input):
        """前向: 阈值二值化"""
        # 确保输入在有效范围
        input = torch.clamp(input, 0.0, 1.0)
        ctx.save_for_backward(input)
        return (input > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向: Straight-Through"""
        return grad_output


class SpikeEncoder(nn.Module):
    """
    脉冲编码器: 将输入图像编码为二值脉冲向量
    
    输出: 
    - firing_probs: 发放概率 π(x) ∈ [0,1]^d
    - 可以从中采样得到二值脉冲 z ∈ {0,1}^d
    
    支持的图像尺寸: 64, 128, 256
    """
    def __init__(self, input_channels=3, latent_dim=20, image_size=64):
        super(SpikeEncoder, self).__init__()
        
        self.image_size = image_size
        
        # 卷积层 (与标准VAE类似)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # BatchNorm for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 计算卷积后的特征图尺寸
        # 每个卷积层: size -> size/2
        # 4层卷积后: image_size / (2^4) = image_size / 16
        conv_output_size = image_size // 16
        flattened_size = 256 * conv_output_size * conv_output_size
        
        # 全连接层: 输出发放概率 π ∈ [0,1]
        # 对于大图像(128+)，添加中间层避免维度跳跃过大
        if image_size >= 128:
            # 添加中间降维层
            hidden_dim = max(512, latent_dim * 4)
            self.fc_hidden = nn.Linear(flattened_size, hidden_dim)
            self.fc_firing_prob = nn.Linear(hidden_dim, latent_dim)
            self.use_hidden = True
        else:
            # 小图像直接连接
            self.fc_firing_prob = nn.Linear(flattened_size, latent_dim)
            self.use_hidden = False
        
    def forward(self, x):
        """
        编码输入为发放概率
        
        Returns:
            firing_probs: 每个隐神经元的发放概率 π(x) ∈ [0,1]^d
        """
        # 卷积编码
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 计算发放概率 (使用sigmoid确保 ∈ [0,1])
        if self.use_hidden:
            # 大图像：使用中间层
            x = F.relu(self.fc_hidden(x))
            # 添加dropout提升稳定性
            x = F.dropout(x, p=0.2, training=self.training)
            firing_probs = torch.sigmoid(self.fc_firing_prob(x))
        else:
            # 小图像：直接连接
            firing_probs = torch.sigmoid(self.fc_firing_prob(x))
        
        # 额外保护：严格限制在[0,1]范围内
        # 防止数值精度问题导致torch.bernoulli报错
        firing_probs = torch.clamp(firing_probs, 0.0, 1.0)
        
        return firing_probs


class SpikeDecoder(nn.Module):
    """
    脉冲解码器: 将二值脉冲向量解码为图像
    
    输入: z ∈ {0,1}^d (二值脉冲)
    输出: 重构图像
    
    支持的图像尺寸: 64, 128, 256
    """
    def __init__(self, latent_dim=20, output_channels=3, image_size=64):
        super(SpikeDecoder, self).__init__()
        
        self.image_size = image_size
        
        # 计算初始特征图尺寸
        self.conv_output_size = image_size // 16
        target_size = 256 * self.conv_output_size * self.conv_output_size
        
        # 全连接层 - 对于大图像添加中间层
        if image_size >= 128:
            hidden_dim = max(512, latent_dim * 4)
            self.fc = nn.Linear(latent_dim, hidden_dim)
            self.fc_expand = nn.Linear(hidden_dim, target_size)
            self.use_hidden = True
        else:
            self.fc = nn.Linear(latent_dim, target_size)
            self.use_hidden = False
        
        # 转置卷积层
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        
        # BatchNorm
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        
    def forward(self, z):
        """
        从二值脉冲解码图像
        
        Args:
            z: 二值脉冲向量 ∈ {0,1}^d
        """
        if self.use_hidden:
            # 大图像：使用中间层
            x = F.relu(self.fc(z))
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.relu(self.fc_expand(x))
        else:
            # 小图像：直接扩展
            x = F.relu(self.fc(z))
        
        x = x.view(x.size(0), 256, self.conv_output_size, self.conv_output_size)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x


class SpikeVAE(nn.Module):
    """
    Spike-VAE: 全脉冲变分自编码器
    
    关键创新:
    1. 隐变量为伯努利分布（而非高斯）
    2. 稀疏脉冲编码（兼具SAE特性）
    3. 使用STE实现端到端训练
    
    支持的图像尺寸: 64, 128, 256
    """
    def __init__(self, input_channels=3, latent_dim=20, prior_sparsity=0.1, image_size=64):
        """
        Args:
            input_channels: 输入通道数
            latent_dim: 隐变量维度（脉冲神经元数量）
            prior_sparsity: 先验发放概率 p (控制稀疏性)
                          - 0.1: 10%的神经元平均激活（高度稀疏）
                          - 0.3: 30%的神经元平均激活（中等稀疏）
                          - 0.5: 50%的神经元平均激活（低稀疏）
            image_size: 输入图像尺寸（64, 128, 或 256）
        """
        super(SpikeVAE, self).__init__()
        
        self.encoder = SpikeEncoder(input_channels, latent_dim, image_size=image_size)
        self.decoder = SpikeDecoder(latent_dim, output_channels=input_channels, image_size=image_size)
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # 先验发放概率 p(z) = Bernoulli(prior_sparsity)
        # 这控制了期望的稀疏性水平
        self.register_buffer('prior_sparsity', torch.tensor(prior_sparsity, dtype=torch.float32))
        
        # STE functions
        self.stochastic_binarize = StochasticBinarize.apply
        self.deterministic_binarize = DeterministicBinarize.apply
        
    def sample_spikes(self, firing_probs, deterministic=False):
        """
        从发放概率采样二值脉冲
        
        Args:
            firing_probs: 发放概率 π ∈ [0,1]
            deterministic: 是否使用确定性采样（推理时）
        
        Returns:
            spikes: 二值脉冲 z ∈ {0,1}
        """
        # 确保firing_probs严格在[0,1]范围内
        # 这是关键修复：防止torch.bernoulli()报错
        firing_probs = torch.clamp(firing_probs, 0.0, 1.0)
        
        if deterministic:
            # 确定性: z = 1 if π > 0.5 else 0
            return self.deterministic_binarize(firing_probs)
        else:
            # 随机: z ~ Bernoulli(π)
            return self.stochastic_binarize(firing_probs)
    
    def forward(self, x, deterministic=False):
        """
        前向传播
        
        Args:
            x: 输入图像
            deterministic: 是否使用确定性采样（推理时使用）
        
        Returns:
            recon_x: 重构图像
            firing_probs: 发放概率 π(x)
            spikes: 二值脉冲 z ∈ {0,1}
        """
        # 编码: 学习发放概率分布 q(z|x) = Bernoulli(π(x))
        firing_probs = self.encoder(x)
        
        # 采样脉冲: z ~ Bernoulli(π)
        spikes = self.sample_spikes(firing_probs, deterministic=deterministic)
        
        # 解码: p(x|z)
        recon_x = self.decoder(spikes)
        
        return recon_x, firing_probs, spikes
    
    def encode(self, x, deterministic=True):
        """编码为脉冲向量"""
        firing_probs = self.encoder(x)
        spikes = self.sample_spikes(firing_probs, deterministic=deterministic)
        return spikes, firing_probs
    
    def decode(self, spikes):
        """从脉冲解码图像"""
        return self.decoder(spikes)
    
    def get_sparsity(self, spikes):
        """计算实际稀疏度（激活神经元的比例）"""
        return spikes.mean()


def bernoulli_kl_divergence(firing_probs, prior_prob):
    """
    伯努利分布的KL散度
    
    KL[Bernoulli(π) || Bernoulli(p)]
    = π*log(π/p) + (1-π)*log((1-π)/(1-p))
    
    Args:
        firing_probs: 后验发放概率 π ∈ [0,1]
        prior_prob: 先验发放概率 p ∈ [0,1] (标量或tensor)
    
    Returns:
        kl: KL散度 (标量)
    """
    # 确保prior_prob在正确的设备上
    if isinstance(prior_prob, torch.Tensor):
        prior_prob = prior_prob.to(firing_probs.device)
    else:
        prior_prob = torch.tensor(prior_prob, dtype=firing_probs.dtype, device=firing_probs.device)
    
    # 避免log(0) - 使用更严格的epsilon
    eps = 1e-7
    firing_probs = torch.clamp(firing_probs, eps, 1.0 - eps)
    
    # 确保prior_prob也是tensor并在有效范围内
    if not isinstance(prior_prob, torch.Tensor):
        prior_prob = torch.tensor(prior_prob, dtype=firing_probs.dtype, device=firing_probs.device)
    prior_prob = torch.clamp(prior_prob, eps, 1.0 - eps)
    
    # KL散度公式 - 使用数值稳定的实现
    # KL = π*log(π/p) + (1-π)*log((1-π)/(1-p))
    # 展开为: π*log(π) - π*log(p) + (1-π)*log(1-π) - (1-π)*log(1-p)
    
    kl_term1 = firing_probs * (torch.log(firing_probs) - torch.log(prior_prob))
    kl_term2 = (1.0 - firing_probs) * (torch.log(1.0 - firing_probs) - torch.log(1.0 - prior_prob))
    kl = kl_term1 + kl_term2
    
    # 检查并处理可能的NaN
    kl = torch.where(torch.isnan(kl), torch.zeros_like(kl), kl)
    kl = torch.where(torch.isinf(kl), torch.zeros_like(kl), kl)
    
    # 对所有维度求和，然后求batch平均
    return kl.sum(dim=1).mean()


def spike_vae_loss(recon_x, x, firing_probs, spikes, prior_sparsity, 
                   beta=1.0, sparsity_weight=0.0):
    """
    Spike-VAE损失函数
    
    Loss = Reconstruction Loss + β * KL Divergence + λ * Sparsity Loss
    
    Args:
        recon_x: 重构图像
        x: 原始图像
        firing_probs: 发放概率 π
        spikes: 二值脉冲 z
        prior_sparsity: 先验发放概率 p
        beta: KL散度权重（β-VAE）
        sparsity_weight: 额外的稀疏性惩罚权重 λ
    
    Returns:
        loss: 总损失
        recon_loss: 重构损失
        kl_loss: KL散度
        sparsity_loss: 稀疏性损失
        actual_sparsity: 实际稀疏度
    """
    batch_size = x.size(0)
    
    # 确保输入和输出在[0,1]范围内（BCE要求）
    # 使用epsilon避免log(0)错误
    eps = 1e-7
    recon_x = torch.clamp(recon_x, eps, 1.0 - eps)
    x = torch.clamp(x, eps, 1.0 - eps)
    
    # 1. 重构损失 (Binary Cross Entropy)
    # E_q(z|x)[log p(x|z)]
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    
    # 2. KL散度 (伯努利分布)
    # KL[q(z|x) || p(z)] where q(z|x) = Bernoulli(π), p(z) = Bernoulli(prior_sparsity)
    kl_loss = bernoulli_kl_divergence(firing_probs, prior_sparsity)
    
    # 3. 额外的稀疏性惩罚 (L1范数)
    # 这鼓励更少的神经元激活（类似SAE）
    sparsity_loss = spikes.mean()  # 平均激活率
    
    # 实际稀疏度（用于监控）
    actual_sparsity = spikes.mean()
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss + sparsity_weight * sparsity_loss
    
    return total_loss, recon_loss, kl_loss, sparsity_loss, actual_sparsity


# ========== 可视化和分析工具 ==========

def analyze_spike_patterns(spikes, firing_probs):
    """
    分析脉冲模式
    
    Returns:
        stats: 统计信息字典
    """
    with torch.no_grad():
        stats = {
            'mean_firing_rate': spikes.mean().item(),  # 平均发放率
            'std_firing_rate': spikes.std().item(),    # 发放率标准差
            'mean_prob': firing_probs.mean().item(),   # 平均发放概率
            'num_active_neurons': (spikes.sum(dim=0) > 0).sum().item(),  # 激活的神经元数
            'sparsity': 1 - spikes.mean().item(),  # 稀疏度
        }
    return stats


def visualize_spike_raster(spikes, title="Spike Raster Plot"):
    """
    可视化脉冲栅格图
    
    Args:
        spikes: [batch_size, latent_dim] 二值脉冲
    """
    import matplotlib.pyplot as plt
    
    spikes_np = spikes.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(spikes_np.T, aspect='auto', cmap='binary', interpolation='nearest')
    plt.xlabel('Sample Index')
    plt.ylabel('Neuron Index')
    plt.title(title)
    plt.colorbar(label='Spike (0 or 1)')
    plt.tight_layout()
    
    return plt.gcf()


# ========== 测试代码 ==========

if __name__ == '__main__':
    print("=" * 60)
    print("Spike-VAE 模型测试")
    print("=" * 60)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64  # 测试图像尺寸
    model = SpikeVAE(
        input_channels=3,
        latent_dim=128,  # 脉冲神经元数量
        prior_sparsity=0.1,  # 期望10%的神经元激活
        image_size=image_size
    ).to(device)
    
    # 测试输入
    batch_size = 8
    x = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    # 前向传播
    print("\n训练模式 (随机采样):")
    model.train()
    recon_x, firing_probs, spikes = model(x, deterministic=False)
    
    print(f"输入形状: {x.shape}")
    print(f"重构形状: {recon_x.shape}")
    print(f"发放概率形状: {firing_probs.shape}")
    print(f"脉冲形状: {spikes.shape}")
    print(f"脉冲取值范围: [{spikes.min():.1f}, {spikes.max():.1f}]")
    
    # 计算损失
    loss, recon_loss, kl_loss, sparsity_loss, actual_sparsity = spike_vae_loss(
        recon_x, x, firing_probs, spikes, 
        prior_sparsity=model.prior_sparsity,
        beta=1.0,
        sparsity_weight=0.01
    )
    
    print(f"\n损失分析:")
    print(f"  总损失: {loss.item():.4f}")
    print(f"  重构损失: {recon_loss.item():.4f}")
    print(f"  KL散度: {kl_loss.item():.4f}")
    print(f"  稀疏性损失: {sparsity_loss.item():.4f}")
    print(f"  实际稀疏度: {actual_sparsity.item():.2%} (期望: 10%)")
    
    # 分析脉冲模式
    stats = analyze_spike_patterns(spikes, firing_probs)
    print(f"\n脉冲模式分析:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 推理模式
    print("\n推理模式 (确定性采样):")
    model.eval()
    with torch.no_grad():
        recon_x_det, firing_probs_det, spikes_det = model(x, deterministic=True)
        print(f"  激活神经元数: {spikes_det.sum(dim=1).mean().item():.1f}/{model.latent_dim}")
        print(f"  稀疏度: {(1 - spikes_det.mean()).item():.2%}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    # 显示模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

