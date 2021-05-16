#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/18 下午9:05
# @Author  : houlinjie
# @Site    : 
# @File    : log_rotate.py.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import logging.handlers
import os
import time
import datetime
import shutil
import glob

global DAY_LOG_DIR
global APP_LOG_DIR
# APP_LOG_DIR = os.environ['MATRIX_APPLOGS_DIR']
APP_LOG_DIR = "./"
LOG_PIPE_PATH = APP_LOG_DIR + "app_log.pipe"

date = datetime.date.today().strftime("%Y%m%d")
DAY_LOG_DIR = APP_LOG_DIR + date
if not os.path.exists(DAY_LOG_DIR):
    os.mkdir(DAY_LOG_DIR)
else:
    pass

#定时整理日志
def log_job():
    app_log_dir = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    log_subffix = '.' + yesterday
    process_str = 'process' + log_subffix
    global DAY_LOG_DIR
    DAY_LOG_DIR = APP_LOG_DIR + app_log_dir
    if not os.path.exists(DAY_LOG_DIR):
        os.mkdir(DAY_LOG_DIR)
    else:
        pass
    if glob.glob(process_str)[0] == "":
        pass
    else:
        shutil.move(glob.glob(process_str)[0], DAY_LOG_DIR)


scheduler = BackgroundScheduler()
scheduler.add_job(log_job, 'cron', hour=1, minute=45)
scheduler.start()

app_log = logging.getLogger("app log")
process_handler = logging.handlers.TimedRotatingFileHandler('process', when='midnight')
app_log.suffix = "%Y-%m-%d"
app_log.setLevel(level=logging.INFO)
app_log.addHandler(process_handler)

rf = os.open(LOG_PIPE_PATH, os.O_RDONLY)
while True:
    log_content = os.read(rf, 4096)
    app_log.info(log_content)