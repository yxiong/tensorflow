#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Aug 10, 2016.

"""Examples for image classification with inception v1 model.

The pre-trained model can be downloaded from https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip.

Usage:
* `run_inference(model_path='/path/to/model/file',
                 label_path='/path/to/label/file',
                 image_path='/path/to/image/file')`
* `trim_graph_model(src_model_path = '/path/to/input/model/file',
                    dst_model_path = '/path/to/output/model/file',
                    output_node = 'output')`
"""


import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from xy_python_utils import image_utils

import pb_file_utils


def prepare_input_data(image):
    as_float = np.asarray(image, "float")
    resized = image_utils.imresize(as_float, [224, 224])
    mean_subtracted = resized - 117
    return np.reshape(mean_subtracted, [1, 224, 224, 3])


def load_model(sess, model_path, output_node = "output"):
    graph_def = sess.graph.as_graph_def()
    with open(model_path, 'r') as f:
        graph_def.ParseFromString(f.read())
    return tf.import_graph_def(graph_def, return_elements = ["input:0", output_node + ":0"])


def trim_graph_model(src_model_path, dst_model_path, output_node, output_node_new_name = "output"):
    """The original model file tensorflow_inception_graph.pb downloaded from Google is quite big (54
    MB), but its size can be significantly reduced if we trim off nodes that are not used for
    inference. Note that GoogleNet has 3 different outlets: "output", "output1", and "output2", with
    "output" the smallest (17 MB) whereas "output2" the most accurate.
    """
    # Extract sub graph.
    graph_def = tf.Graph().as_graph_def()
    with open(src_model_path, 'r') as f:
        graph_def.ParseFromString(f.read())
    sub_graph_def = graph_util.extract_sub_graph(graph_def, [output_node])
    # Rename the output node if necessary.
    if output_node != output_node_new_name:
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            tf.import_graph_def(sub_graph_def, name = "")
            output_tensor = sub_graph.get_tensor_by_name(output_node + ":0")
            tf.identity(output_tensor, name = output_node_new_name)
            sub_graph_def = sub_graph.as_graph_def()
    tf.train.write_graph(sub_graph_def, "", dst_model_path, as_text=False)


def run_inference(model_path, label_path, image_path, output_node = "output"):
    """Run inference on an input image."""
    with tf.Session() as sess:
        input_node, output_node = load_model(sess, model_path, output_node)
        input_data = prepare_input_data(image_utils.imread(image_path))
        prob = sess.run(output_node, {input_node: input_data}).flatten()
    # Read labels.
    with open(label_path, 'r') as f:
        labels = [l.strip() for l in f]
    # Note that prob is a little longer than labels (1008 v.s. 1001).
    label_with_prob = zip(labels, list(prob))
    return sorted(label_with_prob, key = lambda x: x[1], reverse = True)
