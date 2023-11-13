import flickrapi
import urllib
import os

# 设置你的 API 密钥和密钥
api_key = '073fbc75cfda6eaef4555f9ba629da5b'
api_secret = 'fdec7d3620898881'

# 创建 Flickr API 对象
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

# 在 Flickr 上搜索图像
keyword = 'messi'  # 搜索关键字
per_page = 200  # 设置每次下载图片的数量

# 创建一个文件夹来存储下载的图片
download_folder = 'downloaded_images'  # 下载文件夹
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# 获取图片
photos = flickr.photos.search(text=keyword, per_page=per_page, extras='url_o')

for photo in photos['photos']['photo']:
    img_url = photo.get('url_o')

    if img_url:
        folder_name = keyword.replace(' ', '_')  # 使用关键字创建文件夹名
        folder_path = os.path.join(download_folder, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        img_name = f"{photo['id']}.jpg"
        img_path = os.path.join(folder_path, img_name)

        urllib.request.urlretrieve(img_url, img_path)
        print(f"Downloaded: {img_name} to folder: {folder_name}")

print("Download completed.")
