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
from werkzeug import secure_filename

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
UPLOAD_FOLDER = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/static')
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
    logging.info("file_extension: %s", file_extension)
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


def run_graph1(filename, sess):
    with sess.graph.as_default():
        image_width = 256
        image_height = 256
        num_channels = 3
        start_load_graph = time.time()
        y_pred = sess.graph.get_tensor_by_name("y_pred:0")
        ## Let's feed the images to the input placeholders
        x = sess.graph.get_tensor_by_name("x:0")
        # y_true = graph.get_tensor_by_name("y_true:0")
        # y_test_images = np.zeros((1, 2))
        #sess = tf.Session(graph=graph, config=config)
        load_graph_elapsed = time.time() - start_load_graph
        logging.info("load_graph_elapsed: %f:", load_graph_elapsed)

        # compute preprocess image time
        start_process = time.time()
        images = preprocess(os.path.join(UPLOAD_FOLDER, filename), image_height, image_width)
        process_elapsed = time.time() - start_process
        logging.info("process_elapsed: %f:", process_elapsed)

        image = images.eval(session=sess)
        x_batch = image.reshape(1, image_height, image_width, num_channels)
        feed_data_time = time.time()
        ### Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch}
        feed_data_elapsed = time.time() - feed_data_time
        logging.info("feed_data_time:", feed_data_elapsed)
        start_compute_time = time.time()
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        compute_elapsed_time = time.time() - start_compute_time
        logging.info("compute_elapsed_time: %f:", compute_elapsed_time)

        # result is of this format [probabiliy_of_cats probability_of_dogs]
        # print()
        # pred=str(result[0][0]).split(" ")
        # print(pred)
        # os.system("rm /home/houlinjie001/classifi/lianjia/deploy/static/filename")
        #out = {"floorplan": str(result[0][0]), "kitchen": str(result[0][1])}
        """
        out = {"bathroom": str(result[0][0]), "floorplan": str(result[0][1]),
               "livingroom": str(result[0][2])}
        
        out = {"bathroom": str(result[0][0]), "bedroom": str(result[0][1]),
               "floorplan": str(result[0][2]), "kitchen": str(result[0][3]),
               "livingroom": str(result[0][4]), "other": str(result[0][5])}
        """
        #return jsonify(out)
        return result


def run_graph2(filename, sess):
    with sess.graph.as_default():
        image_width = 256
        image_height = 256
        num_channels = 3
        start_load_graph = time.time()
        y_pred = sess.graph.get_tensor_by_name("y_pred:0")
        ## Let's feed the images to the input placeholders
        x = sess.graph.get_tensor_by_name("x:0")
        # y_true = graph.get_tensor_by_name("y_true:0")
        # y_test_images = np.zeros((1, 2))
        # sess = tf.Session(graph=graph, config=config)
        load_graph_elapsed = time.time() - start_load_graph
        logging.info("load_graph_elapsed: %f:", load_graph_elapsed)

        # compute preprocess image time
        start_process = time.time()
        images = preprocess(os.path.join(UPLOAD_FOLDER, filename), image_height, image_width)
        process_elapsed = time.time() - start_process
        logging.info("process_elapsed: %f:", process_elapsed)

        image = images.eval(session=sess)
        x_batch = image.reshape(1, image_height, image_width, num_channels)
        feed_data_time = time.time()
        ### Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch}
        feed_data_elapsed = time.time() - feed_data_time
        logging.info("feed_data_time:", feed_data_elapsed)
        start_compute_time = time.time()
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        compute_elapsed_time = time.time() - start_compute_time
        logging.info("compute_elapsed_time: %f:", compute_elapsed_time)

        # result is of this format [probabiliy_of_cats probability_of_dogs]
        # print()
        # pred=str(result[0][0]).split(" ")
        # print(pred)
        # os.system("rm /home/houlinjie001/classifi/lianjia/deploy/static/filename")
        # out = {"floorplan": str(result[0][0]), "kitchen": str(result[0][1])}
        """
        out = {"bathroom": str(result[0][0]), "floorplan": str(result[0][1]),
               "livingroom": str(result[0][2])}

        out = {"bathroom": str(result[0][0]), "bedroom": str(result[0][1]),
               "floorplan": str(result[0][2]), "kitchen": str(result[0][3]),
               "livingroom": str(result[0][4]), "other": str(result[0][5])}
        """
        # return jsonify(out)
        return result
#merge two model's result



def merge_result(fd, sess1, sess2):

    threads = []
    thread1 = MyThread(run_graph1, args=(fd, sess1))
    thread2 = MyThread(run_graph2, args=(fd, sess2))

    threads.append(thread1)
    threads.append(thread2)

    for thread in threads:
        thread.start()

    thread1.join()
    graph1_res = thread1.get_result()
    logging.info("graph1_res thread1 result!!!!!!!!")
    thread2.join()
    graph2_res = thread2.get_result()
    logging.info("graph2_res thread2 result#########")

    """
    graph1: 
           二分类: floorplan  nonfloorplan 
    graph2:
           五分类：bathroom  bedroom  kitchen  livingroom   other
    
    """
    #判断户型图
    if graph1_res[0][0] > 0.6:
        out = {"floorplan": str(graph1_res[0][0])}
        logging.info("graph1_res[0][0]: %f:", graph1_res[0][0])
        return jsonify(out)
    #判断卫生间
    elif graph1_res[0][1] > 0.6 and graph2_res[0][0] > 0.6:
        out = {"bathroom": str(graph2_res[0][0])}
        logging.info("graph1_res[0][1]: %f:", graph2_res[0][0])
        return jsonify(out)
    #判断卧室
    elif graph1_res[0][1] > 0.6 and  graph2_res[0][1] > 0.6:
        out = {"bedroom": str(graph2_res[0][1])}
        logging.info("graph1_res[0][2]: %f:", graph2_res[0][1])
        return jsonify(out)
    #判断厨房
    elif graph1_res[0][1] > 0.6 and graph2_res[0][2] > 0.6:
        out = {"kitchen": str(graph2_res[0][2])}
        logging.info("graph2_res[0][0]: %f:", graph2_res[0][2])
        return jsonify(out)
    #判断客厅
    elif graph1_res[0][1] > 0.6 and graph2_res[0][3] > 0.6:
        out = {"bedroom": str(graph2_res[0][3])}
        logging.info("graph2_res[0][1]: %f:", graph2_res[0][3])
        return jsonify(out)
    #other
    elif graph1_res[0][1] > 0.6 and graph2_res[0][4] > 0.6:
        out = {"other": str(graph2_res[0][4])}
        logging.info("graph2_res[0][2]: %f:", graph2_res[0][4])
        return jsonify(out)
    #归为非房产类图片，过滤规则设置的比较宽松，如果还不符合就不应该上传
    else:
        out = {"illlegal picture!": str(0)}
        logging.info("illlegal picture!!!")
        return jsonify(out)


app = Flask(__name__)
FLAGS, unparsed = parser.parse_known_args()
g1 = load_graph(FLAGS.graph1)
g2 = load_graph(FLAGS.graph2)
session1 = tf.Session(graph=g1, config=config)
session2 = tf.Session(graph=g2, config=config)

@app.route('/', methods=['POST'])
def parse_request():
    try:
        if request.form['data_type'] == u'1':
            start_request = time.time()
            url = request.form['src_data']
            #parse url return image data
            #传给预处理，经过模型计算返回 分类结果
            fd = url.rsplit('/', 1)[1] + '.1000.jpg'
            start_download = time.time()
            urllibopen(url, UPLOAD_FOLDER, fd)
            download_time_elapsed = time.time() - start_download
            start_classify = time.time()
            ret_value = merge_result(fd, session1, session2)
            #ret_value = run_graph(fd, sess)
            classify_time_elapsed = time.time() - start_classify
            total_elapsed_time = time.time() - start_request
            logging.info("download_time_elapsed: %f", download_time_elapsed)
            logging.info("classify_time_elapsed: %f", classify_time_elapsed)
            logging.info("total_elapsed_time: %f", total_elapsed_time)
            filename = os.path.join(UPLOAD_FOLDER, fd)
            os.remove(filename)
            return ret_value
        elif request.form['data_type'] == u'2':
            #recieve picture data
            #传给预处理，经过模型计算返回分类结果
            start_download = time.time()
            upload_file = request.files['src_data']
            filename = secure_filename(upload_file.filename)
            upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
            download_time_elapsed = time.time() - start_download
            start_classify = time.time()
            ret_value = merge_result(filename, session1, session2)
            logging.info("download_time_elapsed: %f", download_time_elapsed)
            #ret_value = run_graph(filename, sess)
            classify_time_elapsed = time.time() - start_classify
            logging.info("classify_time_elapsed: %f", classify_time_elapsed)
            total_elapsed_time = time.time() - start_download
            logging.info("total_elapsed_time: %f", total_elapsed_time)
            return ret_value

    except Exception as e:
        print e
        return repr(e)
app.run(host="10.200.0.174", port=int("16888"), debug=True, use_reloader=False)

#if __name__ == '__main__':
# linux
"""
http_server = HTTPServer(WSGIContainer(app)) 
http_server.bind(16888, '10.200.0.174') 
http_server.start(num_processes=6) 
IOLoop.instance().start()        
"""
            
