import io
import cv2
import time
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET

# Size limit for our CNN
max_size = 512
count = 0
i = 0

#for i in range(1000):
while count < 1000:
    print('Next address')
    url = urllib.request.urlopen("http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl%20solo&pid=" + str(i)).read()
    tree = ET.parse(io.BytesIO(url))
    root = tree.getroot()

    for child in root:
        file_url = 'http:' + child.attrib['file_url']

        if 'png' not in file_url and 'jpg' not in file_url:
            print("'png' or 'jpg' not in filename")
            continue
        
        try:
            img = urllib.request.urlopen(file_url)
        except:
        	print('RemoteDisconnect')
        	time.sleep(3)
        	continue
        

        img = bytearray(img.read())
        img = np.asarray(img, dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
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

        count += 1
        name = 'imgs/' + str(count) + '.jpg'
        cv2.imwrite(name, cropped_img)

        print('Saved:', count)

    i += 1
