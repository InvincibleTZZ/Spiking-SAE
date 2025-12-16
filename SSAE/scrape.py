import os
import shutil
import requests
import time
from bs4 import BeautifulSoup

# ========== 配置参数 ==========
# 设置最大下载图片数量，设置为 None 表示不限制
MAX_IMAGES = 5000  # 修改这个数字来改变下载数量
# 请求延迟（秒），避免被封IP，建议设置为 0.5-2.0
REQUEST_DELAY = 1.0  # 每次请求之间的延迟
# 请求超时（秒）
REQUEST_TIMEOUT = 30
# ==============================

total_count = 0
root_url = 'http://www.getchu.com'
y_m_url = 'http://www.getchu.com/all/month_title.html'

years = [str(y) for y in list(range(2015, 2020))]
months = [str(m).zfill(2) for m in list(range(1, 13))]

root_dir = './images'
#shutil.rmtree(root_dir, ignore_errors=True)
os.makedirs(root_dir, exist_ok=True)

if MAX_IMAGES:
    print(f"目标下载数量: {MAX_IMAGES} 张图片")
else:
    print("无限制模式：将下载所有可用图片")
print("=" * 60)

payload = {
    'gage': 'all',
    'gc': 'gc' # important
}

types = ['pc_soft', 'dvd_game']

# 设置请求头，模拟浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# 创建 session 以保持 cookies
session = requests.Session()
session.headers.update(headers)

max_retries = 5
download_complete = False

for y in years:
    if download_complete:
        break
    ct = 1
    out_dir = os.path.join(root_dir, y)
    os.makedirs(out_dir, exist_ok=True)
    for m in months:
        if download_complete:
            break
        print("Scraping images in year {}, month {}".format(y, m))
        for t in types:
            if download_complete:
                break
            success = False
            retries = 0
            while not success and not download_complete:
                try:
                    by_year_month_res = session.get(
                        y_m_url, 
                        params={**payload, 'year': y, 'month': m, 'genre': t},
                        timeout=REQUEST_TIMEOUT
                    )
                    by_year_month_res.raise_for_status()  # 检查HTTP状态码
                    
                    year_month_soup = BeautifulSoup(by_year_month_res.text, 'html.parser')
                    game_elems = year_month_soup.find_all('td', class_ = 'dd')
                    
                    if not game_elems:
                        print(f"警告: {y}年{m}月 {t} 类型未找到游戏，可能该时间段没有游戏")
                        success = True
                        continue
                    
                    time.sleep(REQUEST_DELAY)  # 添加延迟
                    for game in game_elems:
                        if download_complete:
                            break
                        game_ref = game.find('a').attrs['href']
                        game_url = root_url + game_ref

                        success = False
                        retries = 0
                        while not success and not download_complete:
                            try:
                                game_page_res = session.get(
                                    game_url, 
                                    params={'gc': 'gc'},
                                    timeout=REQUEST_TIMEOUT
                                )
                                game_page_res.raise_for_status()  # 检查HTTP状态码
                                
                                game_page_soup = BeautifulSoup(game_page_res.text, 'html.parser')
                                img_tags = game_page_soup.find_all('img', attrs = { 'alt': lambda x : x and 'キャラ' in x})
                                
                                if not img_tags:
                                    # 没有找到角色图片，跳过这个游戏
                                    success = True
                                    time.sleep(REQUEST_DELAY)
                                    continue
                                
                                character_tags = [root_url + tag.attrs['src'][1:] for tag in img_tags]
                                for character in character_tags:
                                    # 检查是否达到限制
                                    if MAX_IMAGES and total_count >= MAX_IMAGES:
                                        download_complete = True
                                        break
                                    
                                    try:
                                        # 使用 requests 下载图片，设置 Referer 头
                                        img_headers = headers.copy()
                                        img_headers['Referer'] = game_url  # 关键：设置 Referer 为游戏页面
                                        img_headers['Accept'] = 'image/webp,image/apng,image/*,*/*;q=0.8'
                                        
                                        img_response = session.get(
                                            character,
                                            headers=img_headers,
                                            timeout=REQUEST_TIMEOUT,
                                            stream=True
                                        )
                                        img_response.raise_for_status()
                                        
                                        # 保存图片
                                        img_path = os.path.join(out_dir, '{}_{}.jpg'.format(y, ct))
                                        with open(img_path, 'wb') as f:
                                            for chunk in img_response.iter_content(chunk_size=8192):
                                                f.write(chunk)
                                        
                                        ct += 1
                                        total_count += 1
                                        
                                        # 每下载100张显示一次进度
                                        if total_count % 100 == 0:
                                            print("Total images: {} / {}".format(total_count, MAX_IMAGES if MAX_IMAGES else "∞"))
                                        
                                        time.sleep(REQUEST_DELAY * 0.5)  # 下载图片时稍短的延迟
                                    except requests.exceptions.HTTPError as img_error:
                                        if img_error.response.status_code == 403:
                                            print(f"403 Forbidden: {character}")
                                            print(f"  Referer: {game_url}")
                                        else:
                                            print(f"HTTP错误 {img_error.response.status_code}: {character}")
                                        continue
                                    except Exception as img_error:
                                        print(f"下载图片失败 {character}: {type(img_error).__name__} - {img_error}")
                                        continue
                                    
                                    if MAX_IMAGES and total_count >= MAX_IMAGES:
                                        download_complete = True
                                        break
                                
                                if not download_complete:
                                    print("Total images: {} / {}".format(total_count, MAX_IMAGES if MAX_IMAGES else "∞"))
                                success = True
                                time.sleep(REQUEST_DELAY)  # 请求成功后也添加延迟
                            except requests.exceptions.Timeout as e:
                                print("超时: {} (尝试 {}/{})".format(game_url, retries + 1, max_retries))
                                retries += 1
                                if retries < max_retries:
                                    time.sleep(REQUEST_DELAY * 2)  # 重试前等待更长时间
                                if retries == max_retries:
                                    print("  放弃该URL，继续下一个")
                                    success = True
                            except requests.exceptions.HTTPError as e:
                                print("HTTP错误 {}: {} (尝试 {}/{})".format(e.response.status_code, game_url, retries + 1, max_retries))
                                if e.response.status_code == 403:
                                    print("  403 Forbidden - 可能被网站封禁，建议:")
                                    print("  1. 检查是否需要VPN")
                                    print("  2. 增加REQUEST_DELAY延迟时间")
                                    print("  3. 稍后再试")
                                retries += 1
                                if retries == max_retries:
                                    success = True
                            except requests.exceptions.RequestException as e:
                                print("请求失败 {}: {} (尝试 {}/{})".format(type(e).__name__, game_url, retries + 1, max_retries))
                                retries += 1
                                if retries < max_retries:
                                    time.sleep(REQUEST_DELAY * 2)
                                if retries == max_retries:
                                    success = True
                            except Exception as e:
                                print("未知错误 {}: {} - {}".format(game_url, type(e).__name__, str(e)))
                                retries += 1
                                if retries == max_retries:
                                    success = True
                    success = True
                except requests.exceptions.Timeout as e:
                    print("超时: {} (尝试 {}/{})".format(y_m_url, retries + 1, max_retries))
                    retries += 1
                    if retries < max_retries:
                        time.sleep(REQUEST_DELAY * 3)
                    if retries == max_retries:
                        print("  无法获取该年月数据，跳过")
                        success = True
                except requests.exceptions.HTTPError as e:
                    print("HTTP错误 {}: {} (尝试 {}/{})".format(e.response.status_code, y_m_url, retries + 1, max_retries))
                    if e.response.status_code == 403:
                        print("  403 Forbidden - 可能被网站封禁或需要VPN")
                    retries += 1
                    if retries == max_retries:
                        success = True
                except requests.exceptions.RequestException as e:
                    print("请求失败 {}: {} (尝试 {}/{})".format(type(e).__name__, y_m_url, retries + 1, max_retries))
                    retries += 1
                    if retries < max_retries:
                        time.sleep(REQUEST_DELAY * 3)
                    if retries == max_retries:
                        success = True
                except Exception as e:
                    print("未知错误: {} - {}".format(type(e).__name__, str(e)))
                    retries += 1
                    if retries == max_retries:
                        success = True

print("=" * 60)
print("下载完成！共下载 {} 张图片".format(total_count))
print("图片保存在: {}".format(root_dir))
        


#soup = BeautifulSoup(requests.get('http://www.getchu.com/soft.phtml?id=727363&gc=gc').text, 'html.parser')
#print(soup.find_all('img', attrs = { 'alt': lambda x : x and 'キャラ' in x}))