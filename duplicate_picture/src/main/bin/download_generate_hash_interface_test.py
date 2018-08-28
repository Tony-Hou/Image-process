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


logging.basicConfig(level=logging.INFO)

# 保存异常值的日志文件
abnormal_log = './dup_img_log.txt'
abnormal = open(abnormal_log, 'a+')
log_file = './process_image.txt'
log_fd = open(log_file, 'a+')
# 访问图片前缀
prefix = 'http://image.media.lianjia.com'
# 根据url下载图片

# 保存图像文件名的对立
req_queue = Queue.Queue(256)

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
        logging.info("image url: %s, start process time: %d", img_url, time.time())
        result.add(img_url)
        if img_url == "":
            ret.append({"pic_url": "", "state": 1})
            log_fd.write(
                'request_seq: {} image url : {} start response time: {}\n'.format(
                    request_seq, NUll, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            log_fd.flush()
            return 0
        url = prefix + img_url
        start_response = time.time()
        # 报异常
        response = req.get(url)
        if 200 != response.status_code:
            ret.append({"pic_url": img_url,
                        "state": 2})
            log_fd.write(
                'request_seq: {} no result image url : {} start process time: {}\n'.format(
                    request_seq, img_url, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            log_fd.flush()
            return 0
        image = Image.open(BytesIO(response.content))
        download_time = time.time() - start_response
        logging.info("download_time: %f", download_time)
        # shrink image
        shrink_time = time.time()
        scale_height = int(math.ceil((300 / float(image.size[0])) * image.size[1]))
        pic = image.resize((300, scale_height), Image.BICUBIC)
        shrink_elapsed_time = time.time() - shrink_time
        logging.info("shrink_elapsed_time: %f", shrink_elapsed_time)
        log_fd.write('request_seq: {} image url : {} start download image time: {} download elapsed time: {} shrink_elapsed_time: {}\n'.format(
            request_seq, img_url, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_response)), download_time, shrink_elapsed_time))
        log_fd.flush()
    except Exception as e:
        abnormal.write('request_seq: {} image url: {} download and shrink image error time: {} error detail info: {}\n'.format(request_seq,
                                           img_url, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
        ret.append({"pic_url": img_url,
                   "state": 1})
        return 0
    try:
        gen_hash_start = time.time()
        hash_code = imagehash.phash(pic)
        gen_hash_elapsed_time = time.time() - gen_hash_start
        logging.info('hash code: %s', hash_code)
        ret.append({"pic_url": img_url,
                    "state": 0,
                    "hash_code": str(hash_code)})
        # 记录开始生成hash的时间以及所消耗时间
        log_fd.write('request_seq: {} image url: {} start generate time: {} generate hash elapsed time: {} hash_code: {}\n'.format(
            request_seq, img_url, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(gen_hash_start)),
            gen_hash_elapsed_time, hash_code))
        log_fd.flush()
        logging.info("gen_hash_elapsed_time: %f, request_id %d", gen_hash_elapsed_time, request_id)
        logging.info("image url: %s, process end time: %d", img_url, time.time())
    except Exception as e:
        # 保存异常值到log日志
        abnormal.write('request_seq: {} image url: {} generate hash error time: {} error detail info: {}\n'.format(
                             request_seq, img_url, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
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
    download_pool = threadpool.ThreadPool(16)
    try:
        j_data = json.loads(data)
    except Exception as e:
        abnormal.write('request_num: {} error generate time: {} error info: {}'.format(request_num, time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
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
        abnormal.write('request_num: {} error generate time: {} error info: {}\n'.format(request_num, time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
        ret_val = {
                    "errorCode": 1,
                    "errorMsg": "parameter error"
                   }
        return jsonify(ret_val)
    for img_url in url_list:
        lst_var = [img_url, result_list, ret_value, request_num]
        func_var.append((lst_var, None))
    download_requests = threadpool.makeRequests(download_and_generate_hash, func_var)
    [download_pool.putRequest(req) for req in download_requests]
    download_pool.wait()
    print("debug comment")
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
        abnormal.write('request_id: {} error generate time: {} request data error: {}\n'.format(request_id,
                                                     time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "parameter error"
                    }
        return jsonify(ret_value)
    try:
        j_data = json.loads(data)
    except Exception as e:
        abnormal.write(' request_id: {} error generate time: {} parse json data errors: {}\n'.format(request_id,
                                            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
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
        abnormal.write('request_id: {} parse pic_url error: {} error: {}\n'.format(request_id, time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
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
        image = Image.open(BytesIO(response.content))
        # shrink image
        scale_height = int(math.ceil((300 / float(image.size[0])) * image.size[1]))
        pic = image.resize((300, scale_height), Image.BICUBIC)
        download_time = time.time() - start_response
        # 记录图片开始下载时间以及所消耗时间
        logging.info("download_time: %f", download_time)
        log_fd.write('request_id: {} image url : {} start download image time: {} download elapsed time: {}\n'.format(request_id, url,
                     time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_response)), download_time))
        log_fd.flush()
    except Exception as e:
        abnormal.write('request_id: {} error generate time: {} image url: {} error detail info: {}\n'.format(request_id, time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), url, e))
        abnormal.flush()
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "download image error"
                     }
        return jsonify(ret_value)
    try:
        gen_hash_start = time.time()
        hash_value = imagehash.phash(pic)
        ret_value = {
                       "errorCode": 0,
                       "errorMsg": "success",
                       "pic_url": img_url,
                       "hash_code": str(hash_value)
                     }
        gen_hash_elapsed_time = time.time() - gen_hash_start
        elapsed_time = time.time() - start_parse_request_time
        # 记录开始生成hash的时间以及所消耗时间
        log_fd.write('request_id: {} image url: {} start generate time: {} generate hash elapsed time: {} response '
                     'elapse ''time: {} hash_value: {} \n'.format(request_id, url, time.strftime('%Y-%m-%d %H:%M:%S',
                      time.localtime(gen_hash_start)), gen_hash_elapsed_time, elapsed_time, hash_value))
        log_fd.flush()
        logging.info("gen_hash_elapsed_time: %f, elapsed_time: %f, request_id %d", gen_hash_elapsed_time, elapsed_time,
                     request_id)
    except Exception as e:
        abnormal.write('request_id: {} error time: {} image url: {} error detail info: {}\n'.format(request_id, time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), url, e))
        abnormal.flush()
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
    # 把每次请求放在队列中
        req_queue.put(data)
    except Exception as e:
        abnormal.write('request_id: {} error generate time: {}  error info: {}\n'.format(request_id, time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
        ret_val = {
                     "errorCode": 1,
                     "errorMsg": "get request data error"
                  }
        return jsonify(ret_val)
    try:
        req_data = req_queue.get()
    except Exception as e:
        abnormal.write('request_id: {} error generate time: {} error info: {}\n'.format(request_id, time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time())), e))
        abnormal.flush()
        # 处理请求的数据
        ret_val = {
                     "errorCode": 1,
                     "errorMsg": "get request data error"
                  }
        return jsonify(ret_val)
    hash_code_result = parse_data_task(req_data, request_id)
    elapsed_time = time.time() - start_time
    logging.info("elapsed_time: %s", elapsed_time)
    return hash_code_result


app.run(host="127.0.0.1", port=int("16888"), debug=False, use_reloader=False)
