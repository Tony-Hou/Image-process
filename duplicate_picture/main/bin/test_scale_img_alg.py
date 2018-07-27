#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 上午11:55
# @Author  : houlinjie
# @Site    : 
# @File    : test_scale_img_alg.py
# @Software: PyCharm

import imagehash
import PIL.Image as Image
import time
import os
import logging
import urllib

prefix = 'http://image.media.lianjia.com'
subfix = '!m_fit,w_300,h_300'
# 保存
image_url_file = 'rent_image_url.txt'
url_file = open(image_url_file, 'r')

# 保存hash value的文件
save_hash_result = 'hash_result.txt'
save_hash_fd = open(save_hash_result, 'a+')
# 保存下载下来的原图
original_pic_dir = './ori_pic/'
# 保存下载下来的缩略图
scale_pic_dir = './scale_pic/'

# 保存异常
save_abnormal_file = 'abnormal.txt'
abnormal_file = open(save_abnormal_file, 'a+')

# 根据url下载图片


def urllibopen(url, path, filename):
    try:
        sock = urllib.urlopen(url)
        htmlcode = sock.read()
        sock.close()
        filedir = open(os.path.join(path, filename), "wb")
        filedir.write(htmlcode)
        filedir.close()
    except Exception as err:
        logging.info('open url image error: %s', err)
        abnormal_file.write('{} {} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), url, err))

# 确定保存图片的目录是否已经存在
if not os.path.exists(original_pic_dir):
    os.mkdir(original_pic_dir)

if not os.path.exists(scale_pic_dir):
    os.mkdir(scale_pic_dir)

for line in url_file.readlines():
    # 原图的处理流程
    img_url = line.strip()
    print(img_url)
    ori_filename = img_url.split('/')[-1]
    print(ori_filename)
    url = prefix + img_url
    print(url)
    # download original picture time
    start_download_ori = time.time()
    urllibopen(url, original_pic_dir, ori_filename)
    elapsed_ori_download = time.time() - start_download_ori
    print(elapsed_ori_download)
    # generate original hash value time
    ori_generate_hash_time = time.time()
    try:
        avatar = Image.open(os.path.join(original_pic_dir, ori_filename))
        ori_hash = imagehash.phash(avatar)
    except Exception as err:
        abnormal_file.write('{} {} {} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),img_url, ori_filename, err))
        pass
    ori_elapsed_time = time.time() - ori_generate_hash_time
    # 缩放图片处理过程
    scale_filename = ori_filename + subfix
    scale_url = prefix + img_url + subfix
    # download scale picture time
    start_download_scale = time.time()
    urllibopen(scale_url, scale_pic_dir, scale_filename)
    elapsed_scale_download = time.time() - start_download_scale
    # generate hash value time
    scale_generate_hash_time = time.time()
    try:
        scale_hash = imagehash.phash(Image.open(os.path.join(scale_pic_dir, scale_filename)))
        scale_elapsed_time = time.time() - scale_generate_hash_time
    except Exception as err:
        abnormal_file.write('{} {} {} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            img_url, scale_filename, err))
        pass
    distance = ori_hash - scale_hash
    save_hash_fd.write('{} {} {} {} {} {} {} {} {} {}\n'.format(ori_filename, elapsed_ori_download, ori_elapsed_time,
                                                               elapsed_scale_download, scale_elapsed_time, avatar.size[0],
                                                               avatar.size[1], ori_hash, scale_hash, distance))


# if __name__ == '__main__':
















