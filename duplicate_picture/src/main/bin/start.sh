#!/usr/bin/env bash

#存放代码根目录
#APP_DIR=$MATRIX_CODE_DIR
APP_DIR="/Users/houlinjie/Image-process/duplicate_picture/src/main"
#存放日志的目录
#LOG_PATH=$MATRIX_APPLOGS_DIR
#存放access log日志
#ACCESS_LOGS_PATH=$MATRIX_ACCESSLOGS_DIR
#保存程序配置文件
APP_CONF_DIR="${APP_DIR}/conf"
IP=`ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"`
#gunicorn -b $IP:16888 --workers 8 download_generate_hash_interface:app
nohup gunicorn -b $IP:16888 -c $APP_CONF_DIR/gunicorn_conf.py download_generate_hash_interface_pipe:app &
while true
do
#sleep 1
#echo test
 re=`ps -a|grep download_generate_hash_interface_pipe:app | wc -l`
 if [[ $re -lt 1000 ]]
 then
   sleep 1
 else
  break
 fi
done
python log_rotate.py &
echo "Server startup"


