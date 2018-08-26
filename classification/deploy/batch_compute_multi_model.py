#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:39:19 2017 

@author: houlinjie
"""
from tornado.wsgi import WSGIContainer 
from tornado.httpserver import HTTPServer 
from tornado.ioloop import IOLoop
from flask import Flask, request, jsonify
import json
import urllib
import time
import tensorflow as tf
import os
import logging
import threading
import argparse
import sys



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1

image_width = 256
image_height = 256
num_channels = 3
central_fraction = 0.875

#represent image type jpeg/png
global image_type
image_type = True
"""
server_log = logging.handlers.TimedRotatingFileHandler('server.log', 'D')
server_log.setLevel(logging.DEBUG)
server_log.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'))

error_log = logging.handlers.TimedRotatingFileHandler('error.log', 'D')
error_log.setLevel(logging.ERROR)
error_log.setFormatter(logging.Formatter(
    '%(asctime)s: %(message)s [in %(pathname)s:%(lineno)d]'))

app.logger.addHandler(server_log)
app.logger.addHandler(error_log)
"""

# 获取每个线程的返回值，对threading.Tthread进行封装
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def get_result(self):
        try:
        #如果子线程不使用join function, 此处可能会报没有self.result 的错误
            return self.result
        except Exception:
            return None

    def run(self):
        self.result = self.func(*self.args)





parser = argparse.ArgumentParser()
parser.add_argument(
    '--graph1',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)'
)
parser.add_argument(
    '--graph2',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)'
)

"""
request data form  {'src_data': '', 'data_type': '1'}
src_data: 
         图片的url或者图片数据
data_type:
        1 代表图片的url
"""


"""
    Args: 
         None 
    Returns:
         data format: json string
         out={"image_name": value, "bathroom": value, "bedroom": value, "floorplan": value, "kitchen": value, 
              "livingroom": value, "other": value}
    return example:
                {
                  "bathroom": "0.099628", 
                  "bedroom": "0.434077", 
                  "floorplan": "0.00099361", 
                  "kitchen": "0.14794", 
                  "livingroom": "0.160443", 
                  "other": "0.156919"
                 }
"""
#UPLOAD_FOLDER = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/static')
UPLOAD_FOLDER = "/home/houlinjie001"
logging.basicConfig(level=logging.INFO)


def urllibopen(url, path, filename):
    try:
        sock = urllib.urlopen(url)
        htmlcode = sock.read()
        sock.close()
        filedir = open(os.path.join(path, filename), "wb")
        filedir.write(htmlcode)
        filedir.close()
    except Exception as err:
        logging.info('Url image open error: %s', err)


def load_graph(trained_model):
    """
    method 1: load graph as default graph.
    #Unpersists graph from file as default graph.
    with tf.gfile.GFile(trained_model, 'rb') as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())
         tf.import_graph_def(graph_def, name='')
    """
    #load graph
    with tf.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph
    


# image preprocess
def preprocess(img_name, height, width,
               central_fraction=0.875, scope=None):
    """
    :param image: preprocess image name
    :param height:
    :param width:
    :param central_fraction: fraction of the image to crop
    :param scope: scope  for name_scope
    :return: 3-D float Tensor of prepared image.
    """
    image_raw_data = tf.gfile.FastGFile(img_name, 'r').read()
    file_extension = img_name.rsplit('.', 1)[1]
    #logging.info("file_extension: %s", file_extension)
    if file_extension == 'jpg' or file_extension == 'jpeg':
        image_raw_data = tf.image.decode_jpeg(image_raw_data)
    elif file_extension == 'png':
        image_raw_data = tf.image.decode_png(image_raw_data)
        image_raw_data = tf.image.encode_jpeg(image_raw_data)
        image_raw_data = tf.image.decode_jpeg(image_raw_data)
    #image_raw_data = tf.image.decode_image(image_raw_data)
    image = tf.image.convert_image_dtype(image_raw_data, dtype=tf.uint8)
    if central_fraction:
        image = tf.image.central_crop(image_raw_data, central_fraction=central_fraction)

    if height and width:
        # Resize the image to the specified  height and width
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image






def main():

    def run_graph1(image_string, sess):
        images1 = sess.run(image1, {image_str1: image_string})
        x_batch1 = images1.reshape(1, image_height, image_width, num_channels)
        feed_dict_tmp1 = {x1: x_batch1}
        result = sess.run(y_pred1, feed_dict=feed_dict_tmp1)
        return result

    def run_graph2(image_string, sess):
        images2 = sess.run(image2, {image_str2: image_string})
        x_batch2 = images2.reshape(1, image_height, image_width, num_channels)
        feed_dict_tmp2 = {x2: x_batch2}
        result = sess.run(y_pred2, feed_dict=feed_dict_tmp2)
        return result

    FLAGS, unparsed = parser.parse_known_args()
    g1 = load_graph(FLAGS.graph1)
    g2 = load_graph(FLAGS.graph2)
    session1 = tf.Session(graph=g1, config=config)
    session2 = tf.Session(graph=g2, config=config)

    with g1.as_default():
        # build graph
        image_str1 = tf.placeholder(tf.string)
        if image_type:
            image_raw_data1 = tf.image.decode_jpeg(image_str1, num_channels)
        else:
            image_raw_data1 = tf.image.decode_png(image_str1, num_channels)
        image1 = tf.image.convert_image_dtype(image_raw_data1, dtype=tf.uint8)
        if central_fraction:
            image1 = tf.image.central_crop(image_raw_data1, central_fraction=central_fraction)
        if image_height and image_width:
            # Resize the image to the specified height and width
            image1 = tf.expand_dims(image1, 0)
            image1 = tf.image.resize_bilinear(image1, [image_height, image_width], align_corners=False)
            image1 = tf.squeeze(image1, [0])
        image1 = tf.subtract(image1, 0.5)
        image1 = tf.multiply(image1, 2.0)
        y_pred1 = session1.graph.get_tensor_by_name("y_pred:0")
        x1 = session1.graph.get_tensor_by_name("x:0")
    with g2.as_default():
        # build graph
        image_str2 = tf.placeholder(tf.string)
        if image_type:
            image_raw_data2 = tf.image.decode_jpeg(image_str2, num_channels)
        else:
            image_raw_data2 = tf.image.decode_png(image_str2, num_channels)
        image2 = tf.image.convert_image_dtype(image_raw_data2, dtype=tf.uint8)
        if central_fraction:
            image2 = tf.image.central_crop(image_raw_data2, central_fraction=central_fraction)
        if image_height and image_width:
            # Resize the image to the specified height and width
            image2 = tf.expand_dims(image2, 0)
            image2 = tf.image.resize_bilinear(image2, [image_height, image_width], align_corners=False)
            image2 = tf.squeeze(image2, [0])
        image2 = tf.subtract(image2, 0.5)
        image2 = tf.multiply(image2, 2.0)
        y_pred2 = session2.graph.get_tensor_by_name("y_pred:0")
        x2 = session2.graph.get_tensor_by_name("x:0")

    res = open("predict.txt", 'a+')
    fp = open("tmp.txt", 'r')
    global i
    i = 0

    for fd in fp.readlines():
        start_time = time.time()
        fd = fd.strip()
        with open(os.path.join(UPLOAD_FOLDER, fd)) as f:
            image_string = f.read()
            #多线程
            threads = []
            thread1 = MyThread(run_graph1, args=(image_string, session1))
            thread2 = MyThread(run_graph2, args=(image_string, session2))

            threads.append(thread1)
            threads.append(thread2)
            for thread in threads:
                thread.start()

            # run graph
            thread1.join()
            graph1_res = thread1.get_result()
            thread2.join()
            graph2_res = thread2.get_result()

            """
            graph1: 
                   二分类: floorplan  nonfloorplan 
           graph2:
                   五分类：bathroom  bedroom  kitchen  livingroom   other
            'bathroom': 0, 'bedroom': 1, 'kitchen': 2, 'livingroom': 3, 'other': 4, 'floorplan': 5, 'nonhouse': 6
            """

            #判断户型图
            if graph1_res[0][0] > 0.6:
                #打印当前的处理处理的数量
                res.write('{}   {}\n'.format(fd, 5))
                i = i + 1
                elapsed_time = time.time() - start_time
                sys.stdout.write('\r>> -_- Classify image -_-  %d/%d   elapsed_time  %f' % (i, 7363, elapsed_time))
                sys.stdout.flush()
                continue
            #判断卫生间
            elif graph1_res[0][1] > 0.6 and graph2_res[0][0] > 0.6:
                res.write('{}   {}\n'.format(fd, 0))
                elapsed_time = time.time() - start_time
                i = i + 1
                sys.stdout.write('\r>> -_- Classify image -_-  %d/%d   elapsed_time  %f' % (i, 7363, elapsed_time))
                sys.stdout.flush()
                continue
            #判断卧室
            elif graph1_res[0][1] > 0.6 and graph2_res[0][1] > 0.6:
                res.write('{}  {}\n'.format(fd, 1))
                elapsed_time = time.time() - start_time
                i = i + 1
                sys.stdout.write('\r>> -_- Classify image -_-  %d/%d   elapsed_time  %f' % (i + 1, 7363, elapsed_time))
                sys.stdout.flush()
                continue
            #判断厨房
            elif graph1_res[0][1] > 0.6 and graph2_res[0][2] > 0.6:
                res.write('{}   {}\n'.format(fd, 2))
                elapsed_time = time.time() - start_time
                i = i + 1
                sys.stdout.write('\r>> -_- Classify image -_-  %d/%d   elapsed_time  %f' % (i, 7363, elapsed_time))
                sys.stdout.flush()
                continue
            #判断客厅
            elif graph1_res[0][1] > 0.6 and graph2_res[0][3] > 0.6:
                res.write('{}   {}\n'.format(fd, 3))
                elapsed_time = time.time() - start_time
                i = i + 1
                sys.stdout.write('\r>> -_- Classify image -_-  %d/%d   elapsed_time  %f' % (i, 7363, elapsed_time))
                sys.stdout.flush()
                continue
            #other
            elif graph1_res[0][1] > 0.6 and graph2_res[0][4] > 0.5:
                res.write('{}   {}\n'.format(fd, 4))
                elapsed_time = time.time() - start_time
                i = i + 1
                sys.stdout.write('\r>> -_- Classify image -_-  %d/%d   elapsed_time  %d' % (i, 7363, elapsed_time))
                sys.stdout.flush()
                continue
            #归为非房产类图片，过滤规则设置的比较宽松，如果还不符合就不应该上传
            else:
                res.write('{}   {}\n'.format(fd, 6))
                elapsed_time = time.time() - start_time
                i = i + 1
                sys.stdout.write('\r>> -_- Classify image -_-  %d/%d   elapsed_time  %f' % (i, 7363, elapsed_time))
                sys.stdout.flush()
                continue

if __name__ == "__main__":
    main()
