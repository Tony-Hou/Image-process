import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect,url_for
from werkzeug import secure_filename
from flask import jsonify
import base64
from io import StringIO
import tensorflow as tf 
import numpy as np
import cv2
# Obtain the flask app object
app = flask.Flask(__name__)

UPLOAD_FOLDER = 'static'


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
    image_raw_data_jpg = tf.gfile.FastGFile(img_name, 'rb').read()
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)

    image = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
    if central_fraction:
        image = tf.image.central_crop(img_data_jpg, central_fraction=central_fraction)

    if height and width:
        #Resize the image to the specified  height and width
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def load_graph(trained_model):   
    with tf.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph

@app.route('/')
def index():
    return "Webserver is running"

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        #receive the image file data and save to UPLOAD_FOLDER
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        image_height = 256
        image_width = 256
        num_channels = 3
        #images = []
        # Reading the image using OpenCV
        #image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        """
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        """
        

        graph =app.graph
        y_pred = graph.get_tensor_by_name("y_pred:0")
        ## Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")

        sess = tf.Session(graph=graph)
        images = preprocess(os.path.join(UPLOAD_FOLDER, filename), image_height, image_width)
        print(type(images))
        #print(images.shape())
        image = images.eval(session=tf.Session())
        #x_batch = image.reshape(1, image_height, image_width, num_channels)

        #y_true = graph.get_tensor_by_name("y_true:0") 
        #y_test_images = np.zeros((1, 2))
        x_batch = image.reshape(1, image_height, image_width, num_channels)

        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {x: x_batch}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        # result is of this format [probabiliy_of_cats probability_of_dogs]
        #print()
        #pred=str(result[0][0]).split(" ")
        #print(pred)
        out = {"bathroom": str(result[0][0]), "bedroom": str(result[0][1]),
               "floorplan": str(result[0][2]), "kitchen": str(result[0][3]),
               "livingroom": str(result[0][4]), "other": str(result[0][5])}
        return jsonify(out)
        #return redirect(url_for('just_upload',pic=filename))

    return  '''
    <!doctype html>
    <html lang="en">
    <head>
      <title>Running my first AI Demo</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">
            <nav id="main">
                <a href="http://localhost:5000/demo" >HOME</a>
            </nav>
          <div class="inner cover">

          </div>
          <div class="mastfoot">
          <hr />
            <div class="container">
              <div style="margin-top:5%">
		            <h1 style="color:black">Dogs Cats Classification Demo</h1>
		            <h4 style="color:black">Upload new Image </h4>
		            <form method=post enctype=multipart/form-data>
	                 <p><input type=file name=file>
        	        <input type=submit style="color:black;" value=Upload>
		            </form>
	            </div>	
            </div>
        	</div>
     </div>
   </div>
</body>
</html>

    '''




app.graph=load_graph('./estate_model.pb')
if __name__ == '__main__':
    app.run(host="10.200.0.174", port=int("5000"), debug=True, use_reloader=False)
    
