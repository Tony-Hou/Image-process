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
from werkzeug import secure_filename
import argparse
import sys

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
parser = argparse.ArgumentParser()
parser.add_argument(
    '--graph',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)'
)


image_width = 256
image_height = 256
num_channels = 3
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
        logging.info('Url image open error: %s', e)


def load_graph(trained_model):
    """
    method 1: load graph as default graph.
    #Unpersists graph from file as default graph.
    with tf.gfile.GFile(trained_model, 'rb') as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())
         tf.import_graph_def(graph_def, name='')
    """
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
               central_fraction=0.875):
    """
    :param image: preprocess image name
    :param height:
    :param width:
    :param central_fraction: fraction of the image to crop
    :param scope: scope  for name_scope
    :return: 3-D float Tensor of prepared image.
    """

    image_raw_data = tf.gfile.FastGFile(img_name, 'r').read()
    image_raw_data = tf.image.decode_jpeg(image_raw_data)
    #image_raw_data = tf.image.decode_image(image_raw_data)
    image = tf.image.convert_image_dtype(image_raw_data, dtype=tf.float32)
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



def run_graph(fd, sess2):

    with sess2.as_default() as sess:

        """
            build the graph 
        """
        y_pred = sess.graph.get_tensor_by_name("y_pred:0")
        ## Let's feed the images to the input placeholders
        x = sess.graph.get_tensor_by_name("x:0")
        # y_true = graph.get_tensor_by_name("y_true:0")
        # y_test_images = np.zeros((1, 2))
        #sess = tf.Session(graph=graph, config=config)
        image = preprocess(os.path.join(UPLOAD_FOLDER, fd), image_height, image_width)
        print('-------------------------image_graph---------------:', image.graph)
        print('-------------------------default graph-------------:', tf.get_default_graph())
        image = image.eval(session=sess)
        x_batch = image.reshape(1, image_height, image_width, num_channels)

        feed_data_time = time.time()
        ### Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch}
        feed_data_elapsed = time.time() - feed_data_time
        logging.info("feed_data_time:", feed_data_elapsed)
        start_compute_time = time.time()
        """
            run the graph 
        """
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
        out = {"floorplan": str(result[0][0]), "other": str(result[0][1])}
        """
        out = {"bathroom": str(result[0][0]), "bedroom": str(result[0][1]),
               "kitchen": str(result[0][2]), "livingroom": str(result[0][3]),
               "other": str(result[0][4])}
        #return jsonify(out)

"""
def main():
    FLAGS, unparsed = parser.parse_known_args()
    g1 = load_graph(FLAGS.graph)
    sess = tf.Session(graph=g1, config=config)
    #res = open("predict_label.txt", 'a+')
    fp = open("img_path.txt", 'r')

    i =  0
    for fd in fp.readlines():
        #start_time = time.time()
        fd = fd.strip()
        run_graph(fd, sess)
    sess.close()
"""


app = Flask(__name__)
FLAGS, unparsed = parser.parse_known_args()
g1 = load_graph(FLAGS.graph)
sess1 = tf.Session(graph=g1, config=config)
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
            #
            ret_value = run_graph(fd, sess1)
            classify_time_elapsed = time.time() - start_classify
            total_elapsed_time = time.time() - start_request
            logging.info("download_time_elapsed: %f", download_time_elapsed)
            logging.info("classify_time_elapsed: %f", classify_time_elapsed)
            logging.info("total_elapsed_time: %f", total_elapsed_time)
            filename = os.path.join(UPLOAD_FOLDER, fd)
            os.remove(filename)
            return(ret_value)
        elif request.form['data_type'] == u'2':
            #recieve picture data
            #传给预处理，经过模型计算返回分类结果
            start_download = time.time()
            upload_file = request.files['src_data']
            filename = secure_filename(upload_file.filename)
            upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
            download_time_elapsed = time.time() - start_download
            start_classify = time.time()
            ret_value = run_graph(filename, sess1)
            classify_time_elapsed = time.time() - start_classify
            total_elapsed_time = time.time() - start_download
            logging.info("download_time_elapsed: %f", download_time_elapsed)
            logging.info("classify_time_elapsed: %f", classify_time_elapsed)
            logging.info("total_elapsed_time: %f", total_elapsed_time)
            return(ret_value)

    except Exception as e:
        return repr(e)
app.run(host="10.200.0.174", port=int("16888"), debug=True, use_reloader=False)


"""
http_server = HTTPServer(WSGIContainer(app)) 
http_server.bind(16888, '10.200.0.174') 
http_server.start(num_processes=6) 
IOLoop.instance().start()        
"""


if __name__ == "__main__":
    main()
