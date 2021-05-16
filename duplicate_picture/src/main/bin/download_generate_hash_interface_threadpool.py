#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 下午2:19
# @Author  : houlinjie
# @Site    : 
# @File    : download_generate_hash_interface.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, Response

import os
import sys
import logging
import imagehash
import time
import json
import requests as req
import threading
import math
import Queue
import threadpool


# 保存日志
process_log = logging.getLogger(__name__)
exception_log = logging.getLogger(__name__)
# 设置日志级别
process_log.setLevel(level=logging.DEBUG)
exception_log.setLevel(level=logging.DEBUG)
# 设置格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d -%(message)s')
# FileHandler
#
process_handler = logging.FileHandler('process.log')
exception_handler = logging.FileHandler('exception.log')

process_handler.setFormatter(formatter)
exception_handler.setFormatter(formatter)

process_log.addHandler(process_handler)
exception_log.addHandler(exception_handler)

# 访问图片前缀
prefix = 'http://image.media.lianjia.com'
subprefix = '!m_fit,w_300,h_300'
# 根据url下载图片

# 保存图像文件名的队列
req_queue = Queue.Queue(256)
download_pool = threadpool.ThreadPool(16)
# request id
global request_id
request_id = int(time.time())
q = threading.Lock()


# 解析请求中的数据把数据存放到队列中
# Args:
#     req_data: 请求数据
# Returns:
#        None

def download_and_generate_hash(img_url, result, ret, request_seq):
    try:
        # 判断img_url 是否为空
        start_response_time = time.time()
        process_log.info("img_url: %s, start response time: %s", img_url, time.time())
        result.add(img_url)
        if img_url == "":
            ret.append({"pic_url": "", "state": 1})
            process_log.info('request_seq: %s,image url is: %s', request_seq, NUll)
            return 0
        url = prefix + img_url + subprefix
        start_response = time.time()
        response = req.get(url)
        if 200 != response.status_code:
            ret.append({"pic_url": img_url,
                        "state": 2})
            process_log.info('request_seq: %s, no result image url: %s', request_seq, img_url)
            return 0
        fd = BytesIO(response.content)
        image = Image.open(fd)
        # shrink_time = time.time()
        # scale_height = int(math.ceil((300 / float(image.size[0])) * image.size[1]))
        # pic = image.resize((300, scale_height), Image.BICUBIC)
        # shrink_elapsed_time = time.time() - shrink_time
        download_elapsed_time = time.time() - start_response
        process_log.info('request_seq: %s,image url: %s download elapsed time: %s', request_seq, img_url,
                         download_elapsed_time)
    except Exception as e:
        exception_log.error('request_seq: %s, image url: %s, Error', request_seq, img_url, exc_info=True)
        ret.append({"pic_url": img_url,
                   "state": 1})
        return 0
    try:
        gen_hash_start = time.time()
        hash_code = imagehash.phash(image)
        gen_hash_elapsed_time = time.time() - gen_hash_start
        fd.close()
        image.close()
        # 记录开始生成hash的时间以及所消耗时间
        total_elapsed_time = time.time() - start_response_time
        process_log.info('request_seq: %s, image url: %s, generate hash elapsed time: %s, total_elapsed: %s, hash_code:'
                         '%s', request_seq, img_url, gen_hash_elapsed_time, total_elapsed_time, hash_code)
        ret.append({"pic_url": img_url,
                    "state": 0,
                    "hash_code": str(hash_code)})
    except Exception as e:
        # 保存异常值到log日志
        exception_log.error('request_seq: %s, image url: %s, Error', request_seq, img_url, exc_info=True)
        ret.append({"pic_url": img_url,
                    "state": 1})
        return 0


def parse_data_task(data, request_num):
    # 保存url的列表
    url_list = set([])
    # result_list
    result_list = set([])
    ret_value = []
    func_var = []
    # 如果req_data是无效的json string 直接返回
    try:
        j_data = json.loads(data)
    except Exception as e:
        exception_log.error('request_num: %s, Error', request_num, exc_info=True)
        ret_val = {
                     "errorCode": 1,
                     "errorMsg": "invalid json string"
                  }
        return jsonify(ret_val)
    # 解析json string 过程中发生错误
    try:
        for x in j_data['params']:
            picture_url = x['pic_url']
            url_list.add(picture_url)
    except Exception as e:
        exception_log.error('request_num: %s, Error', request_num, exc_info=True)
        ret_val = {
                    "errorCode": 1,
                    "errorMsg": "parameter error"
                   }
        return jsonify(ret_val)
    try:
        for img_url in url_list:
            lst_var = [img_url, result_list, ret_value, request_num]
            func_var.append((lst_var, None))
        download_requests = threadpool.makeRequests(download_and_generate_hash, func_var)
        [download_pool.putRequest(req) for req in download_requests]
        download_pool.wait()
    except Exception as e:
        exception_log.error('request_num: %s, Error', request_num, exc_info=True)
        ret_val = {
            "errorCode": 1,
            "errorMsg": "parameter error"
        }
        return jsonify(ret_val)
    if url_list.issubset(result_list):
        out = {"errorCode": 0,
               "errorMsg": "success",
               "data": ret_value}
        return json.dumps(out)


app = Flask(__name__)


@app.route('/single-image-process/', methods=['POST', 'GET'])
def single_image_process():
    global request_id
    q.acquire()
    request_id = request_id + 1
    q.release()
    start_parse_request_time = time.time()
    try:
        data = request.get_data()
    except Exception as e:
        exception_log.error('request_id: %s, Error', request_id, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "parameter error"
                    }
        return jsonify(ret_value)
    try:
        j_data = json.loads(data)
    except Exception as e:
        exception_log.error('request_id: %s, Error', request_id, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "cannot parse json string"
                    }
        return jsonify(ret_value)
    try:
        # 如果img_url 的值为NULL,返回错误
        img_url = j_data['pic_url']
        if img_url == "":
            ret_value = {
                "errorCode": 1,
                "errorMsg": "parameter error"
            }
            return jsonify(ret_value)
    except Exception as e:
        exception_log.error('request_id: %s, Error', request_id, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "parameter error"
                    }
        return jsonify(ret_value)
    url = prefix + img_url
    try:
        start_response = time.time()
        response = req.get(url)
        if 200 != response.status_code:
            ret_value = {
                           "errorCode": 2,
                           "errorMsg": "no result picture"
                        }
            return jsonify(ret_value)
        fd = BytesIO(response.content)
        image = Image.open(fd)
        # shrink image
        # scale_height = int(math.ceil((300 / float(image.size[0])) * image.size[1]))
        # pic = image.resize((300, scale_height), Image.BICUBIC)
        download_time = time.time() - start_response
        # 记录图片开始下载时间以及所消耗时间
        process_log.info('request_id: %s, image url: %s, download elapsed time: %s, Error', request_id, url,
                         download_time, exc_info=True)
    except Exception as e:
        exception_log.error('request_id: %s, image url: %s, Error', request_id, url, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "download image error"
                     }
        return jsonify(ret_value)
    try:
        gen_hash_start = time.time()
        hash_value = imagehash.phash(image)
        ret_value = {
                       "errorCode": 0,
                       "errorMsg": "success",
                       "pic_url": img_url,
                       "hash_code": str(hash_value)
                     }
        gen_hash_elapsed_time = time.time() - gen_hash_start
        elapsed_time = time.time() - start_parse_request_time
        # 记录开始生成hash的时间以及所消耗时间
        process_log.info('request_id: %s, image url: %s, generate hash elapsed time: %s, response elapse time: %s,'
                         'hash_value: %s', request_id, img_url, gen_hash_elapsed_time, elapsed_time, hash_value)
        fd.close()
    except Exception as e:
        exception_log.error('request_id: %s, image url: %s, Error', request_id, url, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "generate hash value error"
                     }
        return jsonify(ret_value)
    return jsonify(ret_value)


@app.route('/batch-image-process/', methods=['POST', 'GET'])
def batch_image_process():
    global request_id
    q.acquire()
    request_id = request_id + 1
    q.release()
    try:
        start_time = time.time()
        data = request.get_data()
    except Exception as e:
        exception_log.error('request_id: %s, Error', request_id, exc_info=True)
        ret_val = {
                     "errorCode": 1,
                     "errorMsg": "get request data error"
                  }
        return jsonify(ret_val)
    hash_code_result = parse_data_task(data, request_id)
    elapsed_time = time.time() - start_time
    logging.info("elapsed_time: %s", elapsed_time)
    return hash_code_result


app.run(host="127.0.0.1", port=int("16888"), debug=False, use_reloader=False)
