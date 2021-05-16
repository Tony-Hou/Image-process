#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/18 下午9:53
# @Author  : houlinjie
# @Site    : 
# @File    : client.py.py
# @Software: PyCharm

import os
import logging

APP_LOG_DIR = "./"
LOG_PIPE_PATH = APP_LOG_DIR + "app_log.pipe"
print(LOG_PIPE_PATH)
try:
    os.mkfifo(LOG_PIPE_PATH)
except Exception as e:
    print("mkfifo error %s:", e)


process_log = logging.getLogger('process log')
formatter = logging.Formatter('%(asctime)s -%(message)s')
# process_handler = logging.FileHandler(LOG_PIPE_PATH, os.O_SYNC | os.O_CREAT | os.O_RDWR)
process_handler = logging.FileHandler(LOG_PIPE_PATH)
process_handler.setFormatter(formatter)
process_log.setLevel(level=logging.INFO)
process_log.addHandler(process_handler)

process_log.info("hello world")