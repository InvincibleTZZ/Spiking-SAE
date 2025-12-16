"""
动漫人脸数据集下载脚本
提供多个数据集选项供选择
"""
import os
import argparse
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("提示: 安装 requests 库可以使用自动下载功能: pip install requests")


def download_file(url, save_path, chunk_size=8192):
    """下载文件并显示进度"""
    if not HAS_REQUESTS:
        print("需要安装 requests 库: pip install requests")
        return False
    
    print(f'正在下载: {url}')
    try:
        response = requests.get(url, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f'\r进度: {percent:.1f}%', end='', flush=True)
        print('\n下载完成!')
        return True
    except Exception as e:
        print(f'\n下载失败: {e}')
        return False


def download_animeface_dataset():
    """
    下载AnimeFace Dataset
    这是一个专门用于动漫人脸识别的数据集
    """
    print("=" * 60)
    print("选项1: AnimeFace Dataset")
    print("=" * 60)
    print("这是一个包含约10,000张动漫人脸的数据集")
    print("下载链接（需要手动下载）:")
    print("1. GitHub: https://github.com/Mckinsey666/Anime-Face-Dataset")
    print("2. Kaggle: https://www.kaggle.com/datasets/splcher/animefacedataset")
    print("\n使用方法:")
    print("1. 访问上述链接下载数据集")
    print("2. 解压后将图片放在 data/faces 目录下")
    print("=" * 60)


def download_instructions():
    """显示详细下载说明"""
    print("\n" + "=" * 60)
    print("动漫人脸数据集推荐和下载指南")
    print("=" * 60)
    
    print("\n【推荐数据集】")
    
    print("\n1. AnimeFace Dataset (最推荐)")
    print("- 专门用于动漫人脸识别")
    print("- 包含约10,000张高质量动漫人脸")
    print("- 下载地址:")
    print("  • GitHub: https://github.com/Mckinsey666/Anime-Face-Dataset")
    print("  • Kaggle: https://www.kaggle.com/datasets/splcher/animefacedataset")
    print("  • 直接下载: https://github.com/Mckinsey666/Anime-Face-Dataset/archive/refs/heads/master.zip")
    
    print("\n2. Danbooru2019/2020 (大规模数据集)")
    print("- 包含数百万张动漫图片，需要筛选人脸")
    print("- 下载地址: https://www.gwern.net/Danbooru2020")
    print("- 注意: 数据集很大，需要额外处理提取人脸")
    
    print("\n3. Anime Character Face Dataset")
    print("- Kaggle: https://www.kaggle.com/datasets/subinium/anime-faces")
    print("- 包含多种动漫角色人脸")
    
    print("\n4. 自建数据集")
    print("- 从Pixiv、Danbooru等网站收集动漫人脸图片")
    print("- 使用爬虫工具批量下载")
    
    print("\n" + "=" * 60)
    print("【快速开始 - 使用AnimeFace Dataset】")
    print("=" * 60)
    print("\n方法1: 手动下载")
    print("1. 访问: https://github.com/Mckinsey666/Anime-Face-Dataset")
    print("2. 点击 'Code' -> 'Download ZIP'")
    print("3. 解压后找到图片文件夹")
    print("4. 将所有图片复制到: data/faces/")
    
    print("\n方法2: 使用Git (如果已安装)")
    print("git clone https://github.com/Mckinsey666/Anime-Face-Dataset.git")
    print("然后复制图片到 data/faces/")
    
    print("\n方法3: 使用Kaggle API")
    print("1. 安装: pip install kaggle")
    print("2. 配置Kaggle API密钥")
    print("3. 运行: kaggle datasets download -d splcher/animefacedataset")
    
    print("\n" + "=" * 60)
    print("【数据集准备】")
    print("=" * 60)
    print("下载完成后，确保目录结构如下:")
    print("data/")
    print("  └── faces/")
    print("      ├── image1.jpg")
    print("      ├── image2.png")
    print("      └── ...")
    print("\n注意: ImageFolder会自动处理子目录，所以也可以:")
    print("data/")
    print("  └── faces/")
    print("      └── class1/  (可以是任意名称)")
    print("          ├── image1.jpg")
    print("          └── ...")
    
    print("\n" + "=" * 60)
    print("【最小数据集要求】")
    print("=" * 60)
    print("建议至少准备 1000+ 张图片以获得较好效果")
    print("理想情况下使用 5000+ 张图片")


def create_download_script():
    """创建一个简单的下载脚本"""
    script_content = """#!/usr/bin/env python3
\"\"\"
使用requests下载AnimeFace Dataset的示例脚本
注意: 需要根据实际下载链接修改URL
\"\"\"
import os
import requests
import zipfile
from pathlib import Path

def download_animeface():
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # GitHub下载链接（示例，可能需要更新）
    url = "https://github.com/Mckinsey666/Anime-Face-Dataset/archive/refs/heads/master.zip"
    zip_path = "data/animeface.zip"
    
    print("开始下载...")
    try:
        response = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("下载完成，正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/')
        
        print("解压完成！")
        print("请手动将图片移动到 data/faces/ 目录")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载数据集")

if __name__ == '__main__':
    download_animeface()
"""
    
    with open('download_animeface_simple.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    print("已创建下载脚本: download_animeface_simple.py")


def main():
    parser = argparse.ArgumentParser(description='动漫人脸数据集下载指南')
    parser.add_argument('--show-instructions', action='store_true',
                        help='显示详细下载说明')
    parser.add_argument('--create-script', action='store_true',
                        help='创建下载脚本')
    
    args = parser.parse_args()
    
    if args.show_instructions or (not args.show_instructions and not args.create_script):
        download_instructions()
    
    if args.create_script:
        create_download_script()


if __name__ == '__main__':
    main()

