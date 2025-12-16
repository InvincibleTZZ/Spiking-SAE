import cv2
import sys
import os
import os.path
import shutil
def find_cascade_file():
    """查找 lbpcascade_animeface.xml 文件"""
    possible_paths = [
        "./lbpcascade_animeface.xml",
        "./Anime-Face-Dataset/src/lbpcascade_animeface.xml",
        "../Anime-Face-Dataset/src/lbpcascade_animeface.xml",
        "Anime-Face-Dataset/src/lbpcascade_animeface.xml"
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            return path
    
    raise RuntimeError(
        "未找到 lbpcascade_animeface.xml 文件！\n"
        "请确保该文件存在于以下位置之一：\n"
        "  - ./lbpcascade_animeface.xml\n"
        "  - ./Anime-Face-Dataset/src/lbpcascade_animeface.xml\n"
        "或者从 GitHub 仓库下载该文件。"
    )

def detect(filename, outname, cascade_file = None):
    if cascade_file is None:
        cascade_file = find_cascade_file()
    
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        #print(x, y, w, h)
        cv2.imwrite(outname, image[int(y-0.1*h): int(y+0.9*h), x: x+w])
        return True
    else:
        return False

# ========== 配置参数 ==========
# 输出目录：设置为 'data/faces' 可直接用于训练，或 'cropped' 仅保存裁剪结果
OUTPUT_DIR = 'data/faces'  # 修改这里改变输出目录
# ==============================

# 检查 images 目录是否存在
if not os.path.exists('./images'):
    print("错误: 未找到 ./images 目录！")
    print("请先运行 scrape.py 下载图片。")
    sys.exit(1)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 查找级联分类器文件
cascade_file = find_cascade_file()
print(f"使用级联分类器: {cascade_file}")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 60)

# 获取实际存在的年份目录
existing_years = []
for y in range(2000, 2020):
    img_dir = './images/' + str(y)
    if os.path.exists(img_dir):
        existing_years.append(y)

if not existing_years:
    print("错误: 未找到任何年份的图片目录！")
    print("请先运行 scrape.py 下载图片。")
    sys.exit(1)

print(f"找到以下年份的图片: {existing_years}")
print()

ct = 0
for y in existing_years:
    img_dir = './images/' + str(y)
    files = os.listdir(img_dir)
    print(f"处理年份 {y}，共 {len(files)} 张图片...")
    
    for f in files:
        try:
            output_path = os.path.join(OUTPUT_DIR, '{}_{}.jpg'.format(ct, y))
            if detect(os.path.join(img_dir, f), output_path, cascade_file):
                ct += 1
                if ct % 100 == 0:
                    print(f"已处理 {ct} 张图片...")
        except Exception as e:
            print(f"处理 {f} 时出错: {e}")
            continue

print("=" * 60)
print(f"\n完成！共处理 {ct} 张图片，保存在 {OUTPUT_DIR}/ 目录")
print(f"\n注意:")
print(f"  - 裁剪后的图片尺寸不固定（取决于检测到的人脸大小）")
print(f"  - 训练脚本会自动将所有图片调整到 64x64")
print(f"  - 如果输出目录是 'data/faces'，可以直接运行 train_custom_dataset.py")