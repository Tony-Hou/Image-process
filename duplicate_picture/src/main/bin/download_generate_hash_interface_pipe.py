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
from multiprocessing.dummy import Pool as ThreadPool
from apscheduler.schedulers.background import BackgroundScheduler
import logging.handlers
import os
import sys
import logging
import imagehash
import time
import json
import requests as req
import threading
import math
import datetime
import shutil
import glob

# 访问图片前缀
domain = 'http://image.media.lianjia.com'
subprefix = '!m_fit,w_300,h_300'
AK = 'HT4ASES5HLDBPFKAOEDD'
SK = 'OMws9wMpOfknZm7JLi/zcb6aCEIGVejvneKl0hzp'
# request id
global request_id
request_id = int(time.time())
q = threading.Lock()

global APP_LOG_DIR
# APP_LOG_DIR = os.environ['MATRIX_APPLOGS_DIR']
APP_LOG_DIR = "./"
# ACCESS_LOGS_PATH = os.environ['MATRIX_ACCESSLOGS_DIR']
# named pipe
LOG_PIPE_PATH = APP_LOG_DIR + "app_log.pipe"

#create named pipe
print(LOG_PIPE_PATH)
try:
    os.mkfifo(LOG_PIPE_PATH)
except Exception as e:
    print("mkfifo error: %s", e)

process_log = logging.getLogger('process log')
formatter = logging.Formatter('%(asctime)s -%(message)s')
# process_handler = logging.FileHandler(LOG_PIPE_PATH, os.O_SYNC | os.O_CREAT | os.O_RDWR)
process_handler = logging.FileHandler(LOG_PIPE_PATH)
process_handler.setFormatter(formatter)
process_log.setLevel(level=logging.INFO)
process_log.addHandler(process_handler)


# 解析请求中的数据把数据存放到队列中
# Args:
#     req_data: 请求数据
# Returns:
#        None
def download_and_generate_hash(img_url, result, ret, request_seq):
    try:
        # 判断img_url 是否为空
        start_response_time = time.time()
        process_log.info("img_url: %s|start response time: %s", img_url, time.time())
        result.add(img_url)
        if img_url == "":
            ret.append({"pic_url": "", "state": 1})
            process_log.info('request_seq: %s|image url is: %s', request_seq, '')
            return 0
        url = domain + img_url + subprefix
        start_response = time.time()
        response = req.get(url)
        if 200 != response.status_code:
            ret.append({"pic_url": img_url,
                        "state": 2})
            process_log.info('request_seq: %s|no result image url: %s', request_seq, img_url)
            return 0
        fd = BytesIO(response.content)
        image = Image.open(fd)
        download_elapsed_time = time.time() - start_response
        process_log.info('request_seq: %s|image url: %s|download elapsed time: %s', request_seq, img_url,
                         download_elapsed_time)
    except Exception as e:
        process_log.info('request_seq: %s|image url: %s|Error', request_seq, img_url, exc_info=True)
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
        process_log.info('request_seq: %s|image url: %s|generate hash elapsed time: %s|total_elapsed: %s|hash_code: %s',
                         request_seq, img_url, gen_hash_elapsed_time, total_elapsed_time, hash_code)
        ret.append({"pic_url": img_url,
                    "state": 0,
                    "hash_code": str(hash_code)})
    except Exception as e:
        # 保存异常值到log日志
        process_log.info('request_seq: %s|image url: %s|Error', request_seq, img_url, exc_info=True)
        ret.append({"pic_url": img_url,
                    "state": 1})
        return 0


def parse_data_task(data, request_num):
    # 保存url的列表
    url_list = set([])
    # result_list
    result_list = set([])
    ret_value = []
    download_pool = ThreadPool(16)
    try:
        j_data = json.loads(data)
    except Exception as e:
        process_log.info('request_num: %s|Error', request_num, exc_info=True)
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
        process_log.info('request_num: %s|Error', request_num, exc_info=True)
        ret_val = {
                    "errorCode": 1,
                    "errorMsg": "parameter error"
                   }
        return jsonify(ret_val)
    try:
        for img_url in url_list:
            lst_var = [img_url, result_list, ret_value, request_num]
            download_pool.apply_async(download_and_generate_hash, lst_var)
        download_pool.close()
        download_pool.join()
    except Exception as e:
        process_log.info('request_num: %s|Error', request_num, exc_info=True)
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
        process_log.info('request_id: %s|Error', request_id, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "parameter error"
                    }
        return jsonify(ret_value)
    try:
        j_data = json.loads(data)
    except Exception as e:
        process_log.info('request_id: %s|Error', request_id, exc_info=True)
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
        process_log.info('request_id: %s|Error', request_id, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "parameter error"
                    }
        return jsonify(ret_value)
    url = domain + img_url
    download_url = url + subprefix
    try:
        start_response = time.time()
        response = req.get(download_url)
        if 200 != response.status_code:
            ret_value = {
                           "errorCode": 2,
                           "errorMsg": "no result picture",
                           "pic_url": img_url
                        }
            return jsonify(ret_value)
        fd = BytesIO(response.content)
        image = Image.open(fd)
        download_time = time.time() - start_response
        # 记录图片开始下载时间以及所消耗时间
        process_log.info('request_id: %s|image url: %s|download elapsed time: %s|Error', request_id, url, download_time,
                         exc_info=True)
    except Exception as e:
        process_log.info('request_id: %s|image url: %s|Error', request_id, url, exc_info=True)
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
        process_log.info('REQUEST_ID: %s|client IP: %s|request method: %s|request URL: %s|request parameter: %s|'
                         'HTTP_CODE: %d|response time: %s|hash_value: %s|image url: %s|generate hash elapsed time: %s|',
                         request_id, request.remote_addr, request.method, request.path,j_data, ret_value['errorCode'],
                         elapsed_time, hash_value, img_url, gen_hash_elapsed_time)
        fd.close()
        image.close()
    except Exception as e:
        process_log.info('request_id: %s|image url: %s|Error', request_id, url, exc_info=True)
        ret_value = {
                       "errorCode": 1,
                       "errorMsg": "generate hash value error"
                     }
        return jsonify(ret_value)
    finally:
        process_log.info('REQUEST_ID:%s|client IP:%s|request method:%s|request URL:%s|request parameter:%s|HTTP_CODE%d|'
                         'response time:%s', request_id,request.remote_addr, request.method, request.path, data, 1,
                         elapsed_time)
    return jsonify(ret_value)


@app.route('/batch-image-process/', methods=['POST', 'GET'])
def batch_image_process():
    process_log.info('logging test')
    global request_id
    q.acquire()
    request_id = request_id + 1
    q.release()
    try:
        start_time = time.time()
        data = request.get_data()
    except Exception as e:
        process_log.info('request_id: %s|Error', request_id, exc_info=True)
        ret_val = {
                     "errorCode": 1,
                     "errorMsg": "get request data error"
                  }
        return jsonify(ret_val)
    finally:
        process_log.info('REQUEST_ID:%s|client IP:%s|request method:%s|request URL:%s|request parameter:%s|HTTP_CODE:%d'
                         '|response time:%s', request_id, request.remote_addr, request.method, request.path, data, 1,
                         (time.time() - start_time))
    hash_code_result = parse_data_task(data, request_id)
    elapsed_time = time.time() - start_time
    process_log.info('REQUEST_ID:%s|client IP:%s|request method:%s|request URL:%s|request parameter:%s|HTTP_CODE:%d|'
                     'response time:%s|req_size:%d', request_id, request.remote_addr, request.method, request.path, data,
                     json.loads(hash_code_result)['errorCode'], elapsed_time, len(hash_code_result))
    return hash_code_result


