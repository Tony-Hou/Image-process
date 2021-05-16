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
gevent.monkey.patch_all()

#bind = '127.0.0.1:16888'     #绑定ip和端口号
pidfile = '../log/gunicorn.pid'
workers = multiprocessing.cpu_count() * 2 + 1    #进程数
worker_class = 'gunicorn.workers.ggevent.GeventWorker' #使用gevent模式，还可以使用sync 模式，默认的是sync模式

