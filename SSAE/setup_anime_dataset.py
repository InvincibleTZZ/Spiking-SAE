"""
Anime Face Dataset 自动爬取和设置脚本
基于 https://github.com/bchao1/Anime-Face-Dataset
"""
import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """检查必要的依赖包"""
    required_packages = {
        'requests': 'requests',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'bs4': 'beautifulsoup4'  # scrape.py 需要
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n当前使用的 Python 解释器:")
        print(f"  {sys.executable}")
        print("\n请使用以下命令安装（使用当前 Python 解释器）:")
        print(f"  {sys.executable} -m pip install {' '.join(missing_packages)}")
        print("\n或者如果使用 conda 环境:")
        print(f"  conda install -c conda-forge {' '.join(missing_packages)}")
        return False
    
    print("✓ 所有依赖包已安装")
    return True


def clone_repository():
    """克隆GitHub仓库"""
    repo_url = "https://github.com/bchao1/Anime-Face-Dataset.git"
    repo_dir = "Anime-Face-Dataset"
    
    if os.path.exists(repo_dir):
        print(f"✓ 仓库目录已存在: {repo_dir}")
        print("  如果爬取失败，可以删除此目录后重新克隆")
        return True
    
    print(f"正在克隆仓库: {repo_url}")
    print("注意: 在中国大陆访问 GitHub 可能需要 VPN 或代理")
    print("如果克隆失败，请检查网络连接或使用 VPN")
    print()
    
    try:
        result = subprocess.run(
            ['git', 'clone', repo_url],
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        print("✓ 仓库克隆成功")
        return True
    except subprocess.TimeoutExpired:
        print("✗ 克隆超时（超过60秒）")
        print("\n可能的原因:")
        print("  1. 网络连接不稳定")
        print("  2. 需要 VPN 或代理才能访问 GitHub（中国大陆用户）")
        print("  3. GitHub 服务器响应慢")
        print("\n解决方案:")
        print("  1. 检查网络连接，确保可以访问 GitHub")
        print("  2. 如果在中国大陆，请连接 VPN 后重试")
        print("  3. 或手动下载 ZIP 文件:")
        print("     https://github.com/bchao1/Anime-Face-Dataset/archive/refs/heads/master.zip")
        print("     下载后解压到当前目录，并重命名为 Anime-Face-Dataset")
        return False
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.lower() if e.stderr else ""
        print(f"✗ 克隆失败")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        
        print("\n可能的原因:")
        if "connection" in error_msg or "timeout" in error_msg or "resolve" in error_msg:
            print("  1. 网络连接问题 - 可能需要 VPN 或代理（中国大陆用户）")
            print("  2. 无法访问 GitHub")
        elif "not found" in error_msg or "404" in error_msg:
            print("  1. 仓库地址可能已更改")
        else:
            print("  1. Git 未正确安装或配置")
            print("  2. 网络连接问题")
        
        print("\n解决方案:")
        print("  1. 如果在中国大陆，请连接 VPN 后重试")
        print("  2. 检查 Git 是否正确安装: https://git-scm.com/downloads")
        print("  3. 手动下载 ZIP 文件:")
        print("     https://github.com/bchao1/Anime-Face-Dataset/archive/refs/heads/master.zip")
        print("     下载后解压到当前目录，并重命名为 Anime-Face-Dataset")
        return False
    except FileNotFoundError:
        print("✗ 未找到git命令")
        print("请先安装Git: https://git-scm.com/downloads")
        print("\n或者手动下载仓库ZIP文件:")
        print("  https://github.com/bchao1/Anime-Face-Dataset/archive/refs/heads/master.zip")
        print("  下载后解压到当前目录，并重命名为 Anime-Face-Dataset")
        return False


def run_scrape_script():
    """运行爬取脚本"""
    script_path = "Anime-Face-Dataset/src/scrape.py"
    
    if not os.path.exists(script_path):
        print(f"✗ 脚本不存在: {script_path}")
        return False
    
    print("=" * 60)
    print("开始运行爬取脚本...")
    print("=" * 60)
    print("注意: 爬取过程可能需要较长时间（数小时）")
    print("爬取的图片将保存在: Anime-Face-Dataset/src/images/")
    print("=" * 60)
    
    original_dir = os.getcwd()
    try:
        # 切换到src目录
        os.chdir("Anime-Face-Dataset/src")
        
        # 运行爬取脚本，捕获输出
        result = subprocess.run(
            [sys.executable, 'scrape.py'],
            check=False,
            capture_output=True,
            text=True
        )
        
        # 恢复原目录
        os.chdir(original_dir)
        
        # 显示输出
        if result.stdout:
            print("\n脚本输出:")
            print(result.stdout)
        
        if result.returncode == 0:
            print("✓ 爬取完成")
            return True
        else:
            print("✗ 爬取过程中出现错误")
            if result.stderr:
                print("\n错误信息:")
                print(result.stderr)
            print(f"\n返回码: {result.returncode}")
            print("\n提示: 您可以稍后手动运行: cd Anime-Face-Dataset/src && python scrape.py")
            return False
            
    except Exception as e:
        # 确保恢复原目录
        try:
            os.chdir(original_dir)
        except:
            pass
        print(f"✗ 运行爬取脚本时出错: {e}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False


def run_detect_script():
    """运行人脸检测脚本"""
    script_path = "Anime-Face-Dataset/src/detect.py"
    
    if not os.path.exists(script_path):
        print(f"✗ 脚本不存在: {script_path}")
        return False
    
    print("=" * 60)
    print("开始运行人脸检测脚本...")
    print("=" * 60)
    print("这将从爬取的图片中检测并裁剪出人脸")
    print("处理后的图片将保存在: Anime-Face-Dataset/src/cropped/")
    print("=" * 60)
    
    original_dir = os.getcwd()
    try:
        # 切换到src目录
        os.chdir("Anime-Face-Dataset/src")
        
        # 运行检测脚本，捕获输出
        result = subprocess.run(
            [sys.executable, 'detect.py'],
            check=False,
            capture_output=True,
            text=True
        )
        
        # 恢复原目录
        os.chdir(original_dir)
        
        # 显示输出
        if result.stdout:
            print("\n脚本输出:")
            print(result.stdout)
        
        if result.returncode == 0:
            print("✓ 人脸检测完成")
            return True
        else:
            print("✗ 检测过程中出现错误")
            if result.stderr:
                print("\n错误信息:")
                print(result.stderr)
            print(f"\n返回码: {result.returncode}")
            print("\n提示: 您可以稍后手动运行: cd Anime-Face-Dataset/src && python detect.py")
            return False
            
    except Exception as e:
        # 确保恢复原目录
        try:
            os.chdir(original_dir)
        except:
            pass
        print(f"✗ 运行检测脚本时出错: {e}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False


def copy_to_data_folder():
    """将处理好的图片复制到data/faces目录"""
    source_dir = "Anime-Face-Dataset/src/cropped"
    target_dir = "data/faces"
    
    if not os.path.exists(source_dir):
        print(f"✗ 源目录不存在: {source_dir}")
        print("请先完成爬取和检测步骤")
        return False
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"正在复制图片从 {source_dir} 到 {target_dir}...")
    
    import shutil
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    copied_count = 0
    for file in os.listdir(source_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(target_dir, file)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    
    print(f"✓ 已复制 {copied_count} 张图片到 {target_dir}")
    return True


def main():
    print("=" * 60)
    print("Anime Face Dataset 自动设置脚本")
    print("=" * 60)
    print()
    
    # 步骤1: 检查依赖
    print("步骤1: 检查依赖包...")
    if not check_dependencies():
        return
    print()
    
    # 步骤2: 克隆仓库
    print("步骤2: 克隆GitHub仓库...")
    if not clone_repository():
        return
    print()
    
    # 步骤3: 询问是否运行爬取
    print("步骤3: 运行爬取脚本")
    print("警告: 爬取过程可能需要数小时，且需要网络连接")
    
    # 循环直到用户输入有效的 y 或 n
    while True:
        response = input("是否现在运行爬取脚本? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            response = 'y'
            break
        elif response in ['n', 'no', '否']:
            response = 'n'
            break
        else:
            print("请输入 'y' (是) 或 'n' (否)，不能为空或空格")
    
    if response == 'y':
        if not run_scrape_script():
            print("\n提示: 您可以稍后手动运行: cd Anime-Face-Dataset/src && python scrape.py")
            return
        print()
        
        # 步骤4: 运行检测脚本
        print("步骤4: 运行人脸检测脚本...")
        if not run_detect_script():
            print("\n提示: 您可以稍后手动运行: cd Anime-Face-Dataset/src && python detect.py")
            return
        print()
        
        # 步骤5: 复制到data/faces
        print("步骤5: 复制图片到data/faces目录...")
        copy_to_data_folder()
        print()
        
        print("=" * 60)
        print("✓ 数据集准备完成！")
        print("=" * 60)
        print("现在可以运行训练脚本:")
        print("  python train_custom_dataset.py")
    else:
        print("\n跳过爬取步骤")
        print("您可以稍后手动运行以下命令:")
        print("  cd Anime-Face-Dataset/src")
        print("  python scrape.py")
        print("  python detect.py")
        print("然后运行此脚本的复制功能，或手动复制图片到 data/faces/")


if __name__ == '__main__':
    main()

