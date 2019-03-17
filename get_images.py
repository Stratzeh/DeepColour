import cv2
import requests
import numpy as np

from bs4 import BeautifulSoup

# Size limit for our CNN
max_size = 512
count = 83502

for i in range(1100, 1300):
    url = "http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl%20solo&pid=" + str(i)
    page_response = requests.get(url)
    content = BeautifulSoup(page_response.content, 'lxml')

    posts = content.find_all('post')
    img_urls = [p['file_url'] for p in posts if 'png' in p['file_url'] or 'jpg' in p['file_url']]

    for url in img_urls:
        response = requests.get('http:' + url)
        img = bytearray(response.content)
        img = np.asarray(img, dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        if h >= w:
            scaling = (max_size * 1.0) / w
            img = cv2.resize(img, (int(scaling*w), int(scaling*h)), interpolation=cv2.INTER_CUBIC)
            cropped_img = img[0:max_size, 0:max_size]
        else:
            # w > h
            scaling = (max_size * 1.0) / h
            img = cv2.resize(img, (int(scaling*w), int(scaling*h)), interpolation=cv2.INTER_CUBIC)
            center_w = int(round((w*scaling)/2))
            cropped_img = img[0:max_size, center_w - int(max_size/2):center_w + int(max_size/2)]

        if cropped_img.shape[:2] != (max_size, max_size):
            print("size mismatch, skip")
            continue

        name = 'imgs/' + str(count) + '.jpg'
        cv2.imwrite(name, cropped_img)
        count += 1
        print(count)
