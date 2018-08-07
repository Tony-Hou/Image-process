#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 下午6:35
# @Author  : houlinjie
# @Site    : 
# @File    : download_generate_hash_v1.py
# @Software: PyCharm


import os
import sys
import logging
import urllib
import imagehash
from PIL import Image
import time
from flask import Flask, request, jsonify
import json

logging.basicConfig(level=logging.INFO)

# 保存异常值的日志文件
abnormal_log = './dup_img_log.txt'
abnormal = open(abnormal_log, 'a+')

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
        # logging.info('open url image error: %s', err)
        abnormal.write('{}##{}##{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), url, err))

"""
根据图片生成hash code
Args: 
     img_name
Returns:
     hash_value
"""


def generate_hash(img_name):
    try:
        hash_code = imagehash.phash(Image.open(img_name))
    except Exception as e:
        # 保存异常值到log日志
        logging.info("exception: %s", e)
        abnormal.write('{}##{}##{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                             img_name, e))
    return hash_code

"""
测试用 url example
/rent-house-1/bd1937ff7410e4ded21d1a37aa3c6c3f-1529052493425/d0be6e068172259e4191e7d2b5947ebc.jpg
/rent-user-avatar/db45dc07-1ff5-4b08-836e-5e415d63f20e
"""

if not os.path.exists('./image/'):
    os.mkdir('./image/')

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def parse_request():
    prefix = 'http://image.media.lianjia.com'
    try:
        data = request.get_data()
    except Exception as e:
        abnormal.write('{}##{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        ret_value = {"errorCode": 1,
                     "errorMsg": "parameter error"}
        return jsonify(ret_value)
    try:
        j_data = json.loads(data)
    except Exception as e:
        abnormal.write('{}##{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        ret_value = {"errorCode": 1,
                     "errorMsg": "cannot parse json string"}
        return jsonify(ret_value)
    try:
        img_url = j_data['pic_url']
    except Exception as e:
        abnormal.write('{}##{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        ret_value = {"errorCode": 1,
                     "errorMsg": "parameter error"}
        return jsonify(ret_value)
    url = prefix + img_url
    filename = img_url.split('/')[-1]
    try:
        start_response = time.time()
        urllibopen(url, './image/', filename)
        download_time = time.time() - start_response
        logging.info("download_time: %f", download_time)
    except Exception as e:
        abnormal.write('{}##{}##{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), url, e))
        ret_value = {"errorCode": 1,
                     "errorMsg": "download image error",
                     "pic_url": url,
                     "hash_code": 1
                     }
        return jsonify(ret_value)
    try:
        gen_hash_start = time.time()
        img_path = os.path.join('./image', filename)
        hash_value = generate_hash(img_path)
        # os.remove(img_path)
        ret_value = {"errorCode": 0,
                     "errorMsg": "success",
                     "pic_url": img_url,
                     "hash_code": str(hash_value)
                     }
        # debug comment
        gen_hash_elapsed_time = time.time() - gen_hash_start
        elapsed_time = time.time() - start_response
        logging.info("gen_hash_elapsed_time: %f", gen_hash_elapsed_time)
        logging.info("elapsed_time: %f", elapsed_time)
    except Exception as e:
        abnormal.write('{}##{}##{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                             img_path, e))
        ret_value = {"errorCode": 1,
                     "errorMsg": "generate hash value error",
                     "pic_url": url,
                     "hash_code": 1,
                     }
        return jsonify(ret_value)
    return jsonify(ret_value)


app.run(host="127.0.0.1", port=int("16888"), debug=False, use_reloader=False)
