#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/14 下午3:11
# @Author  : houlinjie
# @Site    : 
# @File    : gunicorn_conf.py.py
# @Software: PyCharm

import logging
import logging.handlers
import os
import multiprocessing
import gevent.monkey
from logging.handlers import WatchedFileHandler
gevent.monkey.patch_all()

# 获取环境变量
# MATRIX_ACCESSLOGS_DIR = "/data0/www/logs/hash.image_quality.rent.ke.com"
# ACCESSLOGS_DIR = os.environ['MATRIX_ACCESSLOGS_DIR']
# ACCESSLOGS_DIR = './access_log'
if not os.path.exists(ACCESSLOGS_DIR):
    os.mkdir(ACCESSLOGS_DIR)
else:
    pass

# log_level = 'info' #日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
# access_log_format = '%(h)s %(t)s %(p)s %(r)s %(s)s %(L)s %(b)s %(f)s %(a)s %(D)s'    #设置gunicorn访问日志格式，错误日志无法设置

# WatchedFileHandler 来记录日志，在server 机器上，凌晨加上一个自动任务，这样日志就能切割了，但是gunicorn
# 的logging 默认使用的是FileHandler, 但是一旦当自动任务备份的时候，它不会自动重新创建，于是便把原有的FileHandler
# 流重定向到了 /dev/null 自己再另外添加想要的Handler即可，
# 保存日志
# loglevel = 'info'

# acclog = logging.getLogger('gunicorn.access')
# app.logger.handlers = acclog.handlers
# app.logger.setLevel(acclog.level)
# 设置格式
# formatter = logging.Formatter(access_log_format)
# acclog.addHandler(WatchedFileHandler('/Users/houlinjie/Image-process/duplicate_picture/src/main/log/gunicorn_access.log'))
# acclog_handler = logging.handlers.TimedRotatingFileHandler('access', when='M', interval=1)
# acclog.suffix = "%Y%m%d-%H%M"
# acclog_handler = logging.handlers.WatchedFileHandler('../log/gunicorn_access.log')
# acclog_handler.setFormatter(formatter)
# acclog.setLevel(level=logging.INFO)
# acclog.addHandler(WatchedFileHandler('../log/gunicorn_access.log'))
# acclog.addHandler(acclog_handler)
# accesslog.propagate = False
# accesslog = '../log/access.log'
bind = '127.0.0.1:16888'     #绑定ip和端口号
# pifile = '../log/gunicorn.pid'
workers = multiprocessing.cpu_count() * 2 + 1    #进程数
worker_class = 'gunicorn.workers.ggevent.GeventWorker' #使用gevent模式，还可以使用sync 模式，默认的是sync模式

