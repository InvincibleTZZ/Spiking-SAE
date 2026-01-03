import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from model_spike_vae import SpikeVAE, analyze_spike_patterns


def load_image(image_path, device, target_size=(64, 64)):
    """加载并预处理图片"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor, image


def visualize_spike_encoding(
    spikes, 
    firing_probs, 
    original_image,
    recon_image,
    save_path=None
):

    fig = plt.figure(figsize=(16, 10))
    
    # 转换为numpy
    spikes_np = spikes.cpu().numpy().squeeze()
    firing_probs_np = firing_probs.cpu().numpy().squeeze()
    
    # 1. 原始图像
    ax1 = plt.subplot(2, 3, 1)
    original_np = original_image.cpu().permute(1, 2, 0).numpy()
    plt.imshow(original_np)
    plt.title('Original Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 2. 重构图像
    ax2 = plt.subplot(2, 3, 2)
    recon_np = recon_image.cpu().squeeze().permute(1, 2, 0).numpy()
    plt.imshow(recon_np)
    plt.title('Reconstructed Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 3. 脉冲栅格图
    ax3 = plt.subplot(2, 3, 3)
    # 重塑为2D显示（如果神经元太多）
    latent_dim = len(spikes_np)
    if latent_dim > 50:
        # 重塑为方形
        size = int(np.ceil(np.sqrt(latent_dim)))
        spikes_2d = np.zeros((size, size))
        spikes_2d.flat[:latent_dim] = spikes_np
        plt.imshow(spikes_2d, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Spike')
    else:
        # 显示为1D条形图
        plt.bar(range(latent_dim), spikes_np, color='black', edgecolor='gray')
        plt.ylim([0, 1.1])
        plt.xlabel('Neuron Index')
        plt.ylabel('Spike (0 or 1)')
    plt.title(f'Spike Pattern\n({spikes_np.sum():.0f}/{latent_dim} active)', 
              fontsize=14, fontweight='bold')
    
    # 4. 发放概率分布
    ax4 = plt.subplot(2, 3, 4)
    plt.hist(firing_probs_np, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Firing Probability')
    plt.ylabel('Count')
    plt.title('Firing Probability Distribution', fontsize=14, fontweight='bold')
    plt.axvline(firing_probs_np.mean(), color='red', linestyle='--', 
                label=f'Mean: {firing_probs_np.mean():.3f}')
    plt.legend()
    
    # 5. 稀疏度分析
    ax5 = plt.subplot(2, 3, 5)
    active_rate = spikes_np.mean()
    sparsity = 1 - active_rate
    
    categories = ['Active\nNeurons', 'Inactive\nNeurons']
    values = [active_rate * 100, sparsity * 100]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    plt.ylabel('Percentage (%)')
    plt.title('Sparsity Analysis', fontsize=14, fontweight='bold')
    plt.ylim([0, 100])
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 6. 神经元激活排序
    ax6 = plt.subplot(2, 3, 6)
    sorted_probs = np.sort(firing_probs_np)[::-1]
    plt.plot(sorted_probs, linewidth=2, color='steelblue')
    plt.xlabel('Neuron Rank')
    plt.ylabel('Firing Probability')
    plt.title('Neurons Ranked by Firing Probability', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加激活阈值线
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'可视化结果已保存: {save_path}')
    
    plt.close()


def inference_spike_vae(
    model_path,
    image_path,
    output_dir='spike_vae_results',
    device='cuda'
):
    """
    使用Spike-VAE对图像进行推理
    
    Args:
        model_path: 模型路径
        image_path: 输入图像路径
        output_dir: 输出目录
        device: 设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print('=' * 60)
    
    # 加载模型
    print(f'加载模型: {model_path}')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    latent_dim = checkpoint.get('latent_dim', 128)
    prior_sparsity = checkpoint.get('prior_sparsity', 0.1)
    image_size = checkpoint.get('image_size', 64)
    
    model = SpikeVAE(
        input_channels=3,
        latent_dim=latent_dim,
        prior_sparsity=prior_sparsity,
        image_size=image_size
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'模型加载完成')
    print(f'  Epoch: {checkpoint.get("epoch", "unknown")}')
    print(f'  隐变量维度: {latent_dim}')
    print(f'  期望稀疏度: {prior_sparsity:.2%}')
    print(f'  训练图像尺寸: {image_size}x{image_size}')
    
    # 加载图像
    print(f'\n加载图像: {image_path}')
    image_tensor, original_image = load_image(image_path, device, target_size=(image_size, image_size))
    
    # 推理
    print('\n执行脉冲编码...')
    with torch.no_grad():
        # 编码为脉冲
        spikes, firing_probs = model.encode(image_tensor, deterministic=True)
        
        # 解码
        recon_image = model.decode(spikes)
    
    # 分析脉冲模式
    stats = analyze_spike_patterns(spikes, firing_probs)
    
    print('\n脉冲编码结果:')
    print('=' * 60)
    print(f'  总神经元数: {latent_dim}')
    print(f'  激活神经元数: {spikes.sum().item():.0f}')
    print(f'  平均发放率: {stats["mean_firing_rate"]:.2%}')
    print(f'  稀疏度: {stats["sparsity"]:.2%}')
    print(f'  平均发放概率: {stats["mean_prob"]:.3f}')
    print('=' * 60)
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存重构图像
    from torchvision.utils import save_image
    recon_path = os.path.join(output_dir, f'{base_name}_reconstructed.png')
    save_image(recon_image, recon_path)
    print(f'\n重构图像已保存: {recon_path}')
    
    # 保存脉冲向量
    spikes_path = os.path.join(output_dir, f'{base_name}_spikes.npy')
    np.save(spikes_path, spikes.cpu().numpy())
    print(f'脉冲向量已保存: {spikes_path}')
    
    # 保存发放概率
    probs_path = os.path.join(output_dir, f'{base_name}_firing_probs.npy')
    np.save(probs_path, firing_probs.cpu().numpy())
    print(f'发放概率已保存: {probs_path}')
    
    # 可视化
    vis_path = os.path.join(output_dir, f'{base_name}_visualization.png')
    visualize_spike_encoding(
        spikes, firing_probs,
        image_tensor.squeeze(),
        recon_image,
        save_path=vis_path
    )
    
    print('\n' + '=' * 60)
    print('推理完成！')
    print('=' * 60)
    
    return {
        'spikes': spikes,
        'firing_probs': firing_probs,
        'reconstructed': recon_image,
        'stats': stats
    }


def batch_inference(
    model_path,
    image_dir,
    output_dir='spike_vae_results_batch',
    device='cuda',
    max_images=10
):
    """
    批量推理多张图像
    """
    # 查找所有图像
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f'错误: 在 {image_dir} 中未找到图像')
        return
    
    image_paths = image_paths[:max_images]
    
    print(f'找到 {len(image_paths)} 张图像，开始批量推理...\n')
    
    all_stats = []
    
    for i, image_path in enumerate(image_paths):
        print(f'\n处理 [{i+1}/{len(image_paths)}]: {os.path.basename(image_path)}')
        print('-' * 60)
        
        result = inference_spike_vae(
            model_path=model_path,
            image_path=image_path,
            output_dir=output_dir,
            device=device
        )
        
        all_stats.append(result['stats'])
    
    # 汇总统计
    print('\n\n' + '=' * 60)
    print('批量推理汇总统计')
    print('=' * 60)
    
    avg_firing_rate = np.mean([s['mean_firing_rate'] for s in all_stats])
    avg_sparsity = np.mean([s['sparsity'] for s in all_stats])
    
    print(f'平均发放率: {avg_firing_rate:.2%}')
    print(f'平均稀疏度: {avg_sparsity:.2%}')
    print('=' * 60)


def main():
    # ========== 配置参数 ==========
    # 模型路径：None表示自动使用最新模型，或指定具体路径
    MODEL_PATH = None  # 例如: 'checkpoints_spike_vae_fixed/spike_vae_epoch_100.pth'
    
    # 输入图像路径（修改这里）
    INPUT_IMAGE = r'D:/lyk/VAE/test_picture.jpg'  # ← 修改为您的测试图片路径
    
    # 输出目录
    OUTPUT_DIR = 'spike_vae_results'
    
    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ==============================
    
    print("=" * 60)
    print("Spike-VAE 推理程序")
    print("=" * 60)
    
    # 查找模型 - 优先查找新的checkpoint目录
    if MODEL_PATH is None:
        # 按优先级查找checkpoint目录
        possible_dirs = [
            'checkpoints_spike_vae_fixed',    # 修复版
            'checkpoints_spike_vae_128',      # 128版本
            'checkpoints_spike_vae',          # 原始版
        ]
        
        checkpoint_dir = None
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                checkpoint_dir = dir_name
                break
        
        if checkpoint_dir is None:
            print(f'错误: 未找到任何checkpoint目录')
            print(f'请先训练模型！')
            return
        
        # 查找所有模型文件（支持不同命名）
        model_files = glob.glob(os.path.join(checkpoint_dir, 'spike_vae*.pth'))
        if not model_files:
            print(f'错误: 在 {checkpoint_dir} 中未找到模型文件')
            return
        
        # 使用最新的模型
        MODEL_PATH = max(model_files, key=os.path.getmtime)
        print(f'\n✓ 找到checkpoint目录: {checkpoint_dir}')
        print(f'✓ 使用最新模型: {os.path.basename(MODEL_PATH)}')
    
    # 检查输入图像
    if not os.path.exists(INPUT_IMAGE):
        print(f'错误: 图像文件不存在: {INPUT_IMAGE}')
        return
    
    # 执行推理
    result = inference_spike_vae(
        model_path=MODEL_PATH,
        image_path=INPUT_IMAGE,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )
    
    # 打印脉冲向量（前20个）
    print('\n脉冲向量（前20个神经元）:')
    spikes_sample = result['spikes'].cpu().numpy().squeeze()[:20]
    print(spikes_sample.astype(int))


if __name__ == '__main__':
    main()

