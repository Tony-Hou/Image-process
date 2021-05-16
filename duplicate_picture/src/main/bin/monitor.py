#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/20 上午12:26
# @Author  : houlinjie
# @Site    : 
# @File    : monitor.py.py
# @Software: PyCharm

import os
import sys
sys.path.append('./api/conf')
import time
import logging
import json
import base64
import ConfigParser

class MyLog(object):
    """
    自定义log类
    """
    def __init__(self, log_file, log_name, level, reciever):
        self.fh = logging.FileHandler(log_file)
        formatter =  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)

        my_level = logging.DEBUG
        if level == "debug":
            my_level = logging.DEBUG
        if level == "info":
            my_level = logging.INFO
        if level == "warning":
            my_level = logging.WARNING
        if level == "error":
            my_level = logging.ERROR

        logger = logging.getLogger(log_name)
        logger.setLevel(my_level)
        logger.addHandler(self.fh)
        self.logger = logger
        self.reciever = reciever

    def __del__(self):
        self.logger.removeHandler(self.fh)
        self.fh.close()

    def debug(self, line):
        """
        debug 模式
        """
        self.logger.debug(line)

    def info(self, line):
        """
        info 模式
        """
        self.logger.info(line)

    def warning(self, line, subject, content, attach_files = None):
        """
        warning 模式
        """
        self.logger.warning(line)
        self.send_mail(subject, content, self.reciever, attach_files)

    def error(self, line):
        """
        error 模式
        """
        self.logger.error(line)

    def send_mail(self, subject, content, reciever, attach_files = None):
        """
        send mail
        """
        if '' in (subject, content, reciever):
            self.error("send mail with bad param")
        else:
            mail_parm_dict = {}
            mail_parm_dict['version'] = "1.0"
            mail_parm_dict['method'] = "mail.sent"
            mail_parm_dict['group'] = "bigdata"
            mail_parm_dict['auth'] = "yuoizSsKggkjOc8vbMwS0OqYHvwTGGbB"
            parm_dict = {}
            parm_dict['to'] = reciever.split(',')
            parm_dict['subject'] = subject
            parm_dict['nick'] = 'noreply'
            parm_dict['body'] = content
            attach_file_dict = {}
            if attach_files != None:
                file_name_list = attach_files.split(',')
                for item in file_name_list:
                    try:
                        fh = open(item, 'r')
                        item_cnt = fh.read()
                        attach_file_dict[item] = base64.b64encode(item_cnt)
                        fh.close()
                    except:
                        continue
            if len(attach_file_dict) > 0:
                parm_dict['attachbody'] = attach_file_dict
            mail_parm_dict['params'] = parm_dict
            mail_parm_dict_str = json.dumps(mail_parm_dict)
            send_mail_cmd = "curl -i -X POST 'http://sms.lianjia.com/lianjia/sms/send' -d '%s'" % mail_parm_dict_str
            ret = os.system(send_mail_cmd)
            if ret != 0:
                self.logger.error("send mail failed")


def init_log(conf_path):
    print conf_path
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    log_file = cf.get('log_info', 'log_file')
    log_name = cf.get('log_info', 'log_name')
    log_level = cf.get('log_info', 'log_level')
    log_wan_reciever = cf.get('log_info', 'log_wan_reciever')
    try:
        my_log = MyLog(log_file, log_name, log_level, log_wan_reciever)
        return my_log
    except:
        sys.stderr.write("failed to create MyLog instance!")
        exit(1)