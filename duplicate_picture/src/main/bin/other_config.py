#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 上午12:23
# @Author  : houlinjie
# @Site    : 
# @File    : other_config.py.py
# @Software: PyCharm

# -*- coding: utf-8 -*-

import ConfigParser
import os
import sys
reload(sys)

# MATRIX_ENV_CONF = os.getenv('MATRIX_ENV_CONF')
MATRIX_ENV_CONF = '/data0/www/htdocs/py2.platform.lianjia.com/system/MATRIX_ENV_CONF.ini'
print "MATRIX_ENV_CONF=%s" % MATRIX_ENV_CONF
# MATRIX_CODE_DIR = os.getenv('MATRIX_CODE_DIR')
MATRIX_CODE_DIR = '/data0/www/htdocs/py2.platform.lianjia.com'
if MATRIX_ENV_CONF is not None:
    config = ConfigParser.ConfigParser()
    config.read(MATRIX_ENV_CONF)
    configs = config.items('base')
    for item in configs:
        key, value = item
        print 'key=%s, value=%s' % (key, value)
        os.environ[key] = value
config = 'config'

port = os.getenv('port')
ipaddr = os.getenv('ipaddr')
matrix_applogs_dir = os.getenv('matrix_applogs_dir')
matrix_accesslogs_dir = os.getenv('matrix_accesslogs_dir')

if ipaddr is None:
    print 'ipaddr is missing'
if port is None:
    print 'port is missing'
if matrix_applogs_dir is None:
    print 'matrix_applogs_dir is missing'
else:
    matrix_applogs_dir = matrix_applogs_dir[1:-1]
if matrix_accesslogs_dir is None:
    print 'matrix_accesslogs_dir is missing'
else:
    matrix_accesslogs_dir = matrix_accesslogs_dir[1:-1]

bind = ipaddr + ":" + port
# The maximum number of pending connections.
backlog = 2048
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 5
threads = 20
worker_class = 'gevent'
# The maximum number of simultaneous clients.
worker_connections = 200
# The maximum number of requests a worker will process before restarting.
max_requests = 5
# Restart workers when code changes.
reload = True
# Chdir to specified directory before apps loading.
chdir = MATRIX_CODE_DIR + "/lib"

raw_env = ['scripts_base_dir=' + MATRIX_CODE_DIR + '/scripts',
           'log_dir=' + matrix_applogs_dir + "/"]
worker_tmp_dir = matrix_applogs_dir
pidfile = os.getenv('matrix_privdata_dir')[1:-1] + '/master.pid'
print "pidfile=%s" % pidfile

accesslog = matrix_applogs_dir + '/access.log'
print "accesslog=%s" % accesslog

access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
# errorlog = matrix_applogs_dir + '/server.log'
errorlog = '/dev/null'
print "errorlog=%s" % errorlog
# debug/info/warning/error/critical
loglevel = 'warning'
# Redirect stdout/stderr to Error log.
capture_output = True

print 'read config done'