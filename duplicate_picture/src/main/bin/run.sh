#!/usr/bin/env bash

#存放代码根目录
#APP_DIR=$MATRIX_CODE_DIR
APP_DIR="/Users/houlinjie/Image-process/duplicate_picture/src/main"
#存放日志的目录
LOG_PATH=$MATRIX_APPLOGS_DIR
#存放access log日志
ACCESS_LOGS_PATH=$MATRIX_ACCESSLOGS_DIR
#保存程序配置文件
APP_CONF_DIR="${APP_DIR}/conf/"

gunicorn -c ../conf/gunicorn_conf.py  download_generate_hash_interface:app
echo "Server startup"

