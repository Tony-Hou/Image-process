import tensorflow as tf
from tensorflow.python.framework import graph_util
import os,sys

output_node_names = "y_pred"
#output_node_names = "Predictions"
saver = tf.train.import_meta_graph('model.ckpt-36861.meta', clear_devices=True)

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
#saver.restore(sess, "./model.ckpt-411125")
saver.restore(sess, tf.train.latest_checkpoint('./'))
#In order to get the prediction of the network, we need to read & pre-process the input
#image in the same way(as training), get hold of y_pred on the graph and pass it the new image in a 
#feed dict. 
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
) 
output_graph="estate_model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
