import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob


def load_image(image_path, device, target_size=(256, 256), return_pil=False):
    """
    加载并预处理单张图片
    
    Args:
        image_path: 图片路径
        device: 设备（CPU或GPU）
        target_size: 目标图像大小
        return_pil: 是否返回PIL图像（用于保存）
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    if return_pil:
        # 返回resize后的PIL图像用于保存
        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image_tensor, image, resized_image
    else:
        return image_tensor, image


def find_latest_model(checkpoint_dir='checkpoints'):
    """查找最新的模型文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有模型文件
    model_files = glob.glob(os.path.join(checkpoint_dir, 'vae_epoch_*.pth'))
    if not model_files:
        return None
    
    # 按修改时间排序，返回最新的
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]


def list_available_checkpoints(checkpoint_dir='checkpoints'):
    """列出所有可用的checkpoint文件及其epoch信息"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    # 查找所有模型文件
    model_files = glob.glob(os.path.join(checkpoint_dir, 'vae_epoch_*.pth'))
    if not model_files:
        return []
    
    checkpoints_info = []
    for model_file in model_files:
        try:
            # 从文件名提取epoch号
            filename = os.path.basename(model_file)
            epoch_num = int(filename.replace('vae_epoch_', '').replace('.pth', ''))
            
            # 尝试加载checkpoint获取更多信息
            try:
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                loss = checkpoint.get('loss', 'N/A')
                image_size = checkpoint.get('image_size', 'N/A')
                checkpoints_info.append({
                    'epoch': epoch_num,
                    'path': model_file,
                    'loss': loss,
                    'image_size': image_size,
                    'mtime': os.path.getmtime(model_file)
                })
            except:
                checkpoints_info.append({
                    'epoch': epoch_num,
                    'path': model_file,
                    'loss': 'N/A',
                    'image_size': 'N/A',
                    'mtime': os.path.getmtime(model_file)
                })
        except:
            continue
    
    # 按epoch排序
    checkpoints_info.sort(key=lambda x: x['epoch'])
    return checkpoints_info


def find_model_by_epoch(checkpoint_dir='checkpoints', epoch=None):
    """根据epoch号查找模型文件"""
    if epoch is None:
        return find_latest_model(checkpoint_dir)
    
    model_path = os.path.join(checkpoint_dir, f'vae_epoch_{epoch}.pth')
    if os.path.exists(model_path):
        return model_path
    else:
        return None


def inference_single_image(model_path, image_path, output_path, encoded_image_path=None, device='cuda', image_size=256, noise_std=0.1):
    """
    对单张图片进行VAE编码-解码
    
    Args:
        model_path: 训练好的模型路径
        image_path: 输入图片路径（可以是任意尺寸，会自动resize到训练尺寸）
        output_path: 解码输出图片保存路径
        encoded_image_path: 编码输入图片保存路径（resize后的图片）
        device: 设备（'cuda'或'cpu'）
        image_size: 图像尺寸（需要与训练时一致）
        noise_std: 高斯噪声的标准差（0表示不加噪声，建议范围0.05-0.3）
    """
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'模型训练尺寸: {image_size}x{image_size}')
    print('=' * 60)
    
    # 加载模型
    print(f'加载模型: {model_path}')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从checkpoint获取配置，如果没有则使用默认值
    latent_dim = checkpoint.get('latent_dim', 20)
    checkpoint_image_size = checkpoint.get('image_size', None)
    
    # 如果checkpoint中有image_size，使用它（这是训练时的尺寸）
    if checkpoint_image_size:
        image_size = checkpoint_image_size
        print(f'从checkpoint读取训练尺寸: {image_size}x{image_size}')
    
    # 创建模型实例（根据图像尺寸选择模型）
    if image_size > 64:
        from model_highres import VAE
        model = VAE(input_channels=3, latent_dim=latent_dim, image_size=image_size).to(device)
    else:
        from model import VAE
        model = VAE(input_channels=3, latent_dim=latent_dim).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'模型加载完成，epoch: {checkpoint.get("epoch", "unknown")}')
    print(f'隐变量维度: {latent_dim}')
    
    # 加载输入图片
    print(f'\n加载原始图片: {image_path}')
    if not os.path.exists(image_path):
        print(f'错误: 图片文件不存在: {image_path}')
        return
    
    # 加载原始图片并获取尺寸信息
    original_pil = Image.open(image_path).convert('RGB')
    original_size = original_pil.size
    print(f'原始图片尺寸: {original_size[0]}x{original_size[1]}')
    
    # 将图片resize到训练尺寸（用于编码）
    print(f'将图片resize到训练尺寸: {image_size}x{image_size}')
    input_tensor, _, resized_image = load_image(image_path, device, target_size=(image_size, image_size), return_pil=True)
    
    # ========== 添加高斯噪声 ==========
    if noise_std > 0:
        print(f'\n添加高斯噪声 (std={noise_std})...')
        # 生成高斯噪声（与input_tensor相同的形状）
        noise = torch.randn_like(input_tensor) * noise_std
        # 添加噪声并限制在[0,1]范围内
        noisy_tensor = torch.clamp(input_tensor + noise, 0.0, 1.0)
        
        # 保存加噪后的图片
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        noisy_image_path = f'{base_name}_noisy_input_{image_size}x{image_size}_std{noise_std:.2f}.png'
        from torchvision.utils import save_image
        save_image(noisy_tensor, noisy_image_path)
        print(f'保存加噪后的输入图片: {noisy_image_path}')
        print(f'  - 尺寸: {image_size}x{image_size}')
        print(f'  - 噪声标准差: {noise_std}')
        
        # 使用加噪后的tensor进行推理
        input_tensor = noisy_tensor
    else:
        noisy_image_path = None
        print('\n未添加噪声（noise_std=0）')
    # ==================================
    
    # 保存resize后的编码输入图片（原始，未加噪）
    if encoded_image_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        encoded_image_path = f'{base_name}_encoded_input_{image_size}x{image_size}.png'
    
    print(f'\n保存编码输入图片（resize后，原始）: {encoded_image_path}')
    resized_image.save(encoded_image_path)
    print(f'  - 尺寸: {image_size}x{image_size}')
    
    # 推理
    print('\n正在进行编码-解码...')
    with torch.no_grad():
        # 编码
        mu, logvar = model.encoder(input_tensor)
        
        # 重参数化采样
        z = model.reparameterize(mu, logvar)
        
        # 解码
        output_tensor = model.decoder(z)
    
    # 保存解码输出图片
    print(f'\n保存解码输出图片: {output_path}')
    from torchvision.utils import save_image
    save_image(output_tensor, output_path)
    print(f'  - 尺寸: {image_size}x{image_size}')
    
    print('=' * 60)
    print('完成！')
    print(f'原始输入图片: {image_path} ({original_size[0]}x{original_size[1]})')
    print(f'编码输入图片（原始）: {encoded_image_path} ({image_size}x{image_size}) - 已resize到训练尺寸')
    if noisy_image_path:
        print(f'编码输入图片（加噪）: {noisy_image_path} ({image_size}x{image_size}) - 噪声std={noise_std}')
    print(f'解码输出图片: {output_path} ({image_size}x{image_size})')
    print(f'\n隐变量信息:')
    print(f'  维度: {latent_dim}')
    print(f'  均值 (前5个): {mu.squeeze().cpu().numpy()[:5]}')
    print(f'  标准差 (前5个): {torch.exp(0.5 * logvar).squeeze().cpu().numpy()[:5]}')


def main():
    # ========== 配置参数 ==========
    # 输入图片路径
    INPUT_IMAGE = r'D:\lyk\VAE\test_picture.jpg'
    
    # 输出图片路径（如果为None，则自动生成）
    OUTPUT_IMAGE = None  # 例如: 'output.png' 或 None（自动生成）
    
    # 模型选择方式：
    # - None: 自动使用最新的模型
    # - 数字（如 50）: 使用指定epoch的模型
    # - 字符串路径: 直接指定模型路径（如 'checkpoints/vae_epoch_50.pth'）
    SELECTED_EPOCH = 95  # 例如: 50 表示使用 epoch 50 的模型，或 None 使用最新模型
    
    # 图像尺寸（如果为None，则从模型checkpoint中读取）
    IMAGE_SIZE = 256
    
    # 高斯噪声标准差（0表示不加噪声，建议范围0.05-0.3）
    # 0.05: 轻微噪声，0.1: 中等噪声，0.2-0.3: 较强噪声
    NOISE_STD = 1  # 修改这里控制噪声强度
    # ==============================
    
    print("=" * 60)
    print("VAE 图片推理")
    print("=" * 60)
    
    # 列出所有可用的checkpoint
    checkpoints = list_available_checkpoints()
    
    if not checkpoints:
        print('错误: 未找到任何模型文件！')
        print('请确保 checkpoints/ 目录下有训练好的模型')
        return
    
    
    
    # 选择模型
    if SELECTED_EPOCH is None:
        # 自动选择最新的
        latest_cp = max(checkpoints, key=lambda x: x['epoch'])
        MODEL_PATH = latest_cp['path']
        print(f'\n使用最新的模型: Epoch {latest_cp["epoch"]}')
    elif isinstance(SELECTED_EPOCH, int):
        # 根据epoch选择
        MODEL_PATH = find_model_by_epoch(epoch=SELECTED_EPOCH)
        if MODEL_PATH is None:
            print(f'\n错误: 未找到 Epoch {SELECTED_EPOCH} 的模型文件')
            print(f'可用的epoch: {[cp["epoch"] for cp in checkpoints]}')
            return
        print(f'\n使用模型: Epoch {SELECTED_EPOCH}')
    else:
        # 直接使用指定路径
        MODEL_PATH = SELECTED_EPOCH
        if not os.path.exists(MODEL_PATH):
            print(f'错误: 模型文件不存在: {MODEL_PATH}')
            return
        print(f'\n使用指定模型: {MODEL_PATH}')
    
    # 检查输入图片
    if not os.path.exists(INPUT_IMAGE):
        print(f'\n错误: 输入图片不存在: {INPUT_IMAGE}')
        print(f'绝对路径: {os.path.abspath(INPUT_IMAGE)}')
        
        # 检查目录是否存在
        dir_path = os.path.dirname(INPUT_IMAGE)
        if os.path.exists(dir_path):
            print(f'\n目录存在: {dir_path}')
            print('目录中的图片文件:')
            try:
                files = os.listdir(dir_path)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
                if image_files:
                    for img_file in sorted(image_files):
                        full_path = os.path.join(dir_path, img_file)
                        exists = "✓" if os.path.exists(full_path) else "✗"
                        print(f'  {exists} {img_file}')
                else:
                    print('  (未找到图片文件)')
            except Exception as e:
                print(f'  无法列出目录内容: {e}')
        else:
            print(f'\n目录不存在: {dir_path}')
        
        # 尝试查找相似的文件名（不区分大小写）
        if os.path.exists(dir_path):
            filename = os.path.basename(INPUT_IMAGE)
            filename_lower = filename.lower()
            try:
                files = os.listdir(dir_path)
                similar_files = [f for f in files if f.lower() == filename_lower]
                if similar_files and similar_files[0] != filename:
                    print(f'\n提示: 找到相似文件名（可能是大小写问题）:')
                    for f in similar_files:
                        print(f'  {f}')
                    print(f'\n建议使用: {os.path.join(dir_path, similar_files[0])}')
            except:
                pass
        
        return
    
    # 自动生成输出路径
    if OUTPUT_IMAGE is None:
        base_name = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
        OUTPUT_IMAGE = f'{base_name}_decoded_output.png'
    
    # 从checkpoint读取训练时的图像尺寸（如果IMAGE_SIZE为None）
    if IMAGE_SIZE is None:
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            IMAGE_SIZE = checkpoint.get('image_size', 256)  # 默认256
            print(f'从模型读取训练尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}')
        except Exception as e:
            IMAGE_SIZE = 256  # 默认256
            print(f'无法读取模型配置，使用默认尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}')
            print(f'错误信息: {e}')
    else:
        print(f'使用指定的图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}')
        print('注意: 如果与模型训练尺寸不一致，结果可能不理想')
    
    print()
    
    # 执行推理
    inference_single_image(
        model_path=MODEL_PATH,
        image_path=INPUT_IMAGE,
        output_path=OUTPUT_IMAGE,
        encoded_image_path=None,  # 自动生成编码输入图片路径
        device='cuda',
        image_size=IMAGE_SIZE,
        noise_std=NOISE_STD  # 高斯噪声标准差
    )


if __name__ == '__main__':
    main()

