import tensorflow as tf
from tensorflow.python.framework import graph_util
import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	'--meta',
	required=True,
	type=str,
	help='input model checkpoint meta data file (.meta)'
	)
parser.add_argument(
	'--prefix',
	required=True,
	type=str,
	help='input model data prefix')
FLAGS, unparsed = parser.parse_known_args() 

output_node_names = "y_pred"
#saver = tf.train.import_meta_graph('model.ckpt-74928.meta', clear_devices=True)
saver = tf.train.import_meta_graph(FLAGS.meta, clear_devices=True)

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
#saver.restore(sess, "./model.ckpt-74928")
saver.restore(sess, FLAGS.prefix)
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
) 
output_graph="estate_model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
