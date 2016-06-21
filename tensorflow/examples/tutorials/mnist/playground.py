#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Jun 20, 2016.

"""Export trained model to frozen protobuf format.

Usage:
* `python fully_connected_feed.py` to download data (result in `data/` folder) and create a training model (result in `train/` folder).
* `python -c "import playground; playground.play_export_frozen_model()"` to freeze trained parameter into graph (result in `train/frozen-graph.pb`).
* `python -c "import playground; playground.play_run_inference(use_frozen_graph = True/False)"` to run inference on test data.
* `python -c "import playground; playground.play_visualize_graph(frozen = True/False)"` to visualize the graph (result in `visualize` folder), run `tensorboard --logdir=visualize` to visualize.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from fully_connected_feed import placeholder_inputs
import input_data
import mnist


def load_test_data(data_dir = "data"):
    data_sets = input_data.read_data_sets(data_dir, False)
    return data_sets.test


def build_graph(sess,
               hidden1 = 128,
               hidden2 = 32,
               batch_size = 1):
    input_node = tf.placeholder(tf.float32,
                                shape=(batch_size, mnist.IMAGE_PIXELS),
                                name = "input")
    logits = mnist.inference(input_node,
                             hidden1,
                             hidden2)
    output_node = tf.identity(logits, name = "output")
    return input_node, output_node


def load_model(sess, model_path = "train/model-1999"):
    saver = tf.train.Saver()
    saver.restore(sess, model_path)


def freeze_model(sess,
                 model_file = "train/model-1999",
                 graph_file = "train/graph.pb",
                 frozen_graph_file = "train/frozen-graph.pb"):
    tf.train.write_graph(sess.graph_def, "", graph_file, as_text=True)
    saver = tf.train.Saver()
    freeze_graph.freeze_graph(
        input_graph = graph_file,
        input_saver = "",
        input_binary = False,
        input_checkpoint = model_file,
        output_node_names = "output",
        restore_op_name = "save/restore_all",
        filename_tensor_name = "save/Const:0",
        output_graph = frozen_graph_file,
        clear_devices = False,
        initializer_nodes = "")


def load_frozen_model(sess,
                      frozen_graph_file = "train/frozen-graph.pb"):
    with open(frozen_graph_file, "rb") as f:
        graph_def = sess.graph.as_graph_def()
        graph_def.ParseFromString(f.read())
    return tf.import_graph_def(
        graph_def,
        return_elements = ["input:0", "output:0"])


def run_inference(sess, input_node, output_node, input_data):
    logits = sess.run(output_node, {input_node: input_data})
    return np.argmax(logits, axis=1)


def play_export_frozen_model():
    with tf.Session() as sess:
        build_graph(sess)
        freeze_model(sess)


def play_run_inference(use_frozen_graph = True):
    test_data = load_test_data()
    with tf.Session() as sess:
        if use_frozen_graph:
            input_node, output_node = load_frozen_model(sess)
        else:
            input_node, output_node = build_graph(sess)
            load_model(sess)
        for i in xrange(10):
            input_data = np.atleast_2d(test_data.images[i])
            prediction = run_inference(sess, input_node, output_node, input_data)
            print "Prediction: %d. Ground truth: %d." % (prediction[0], test_data.labels[i])


def play_visualize_graph(frozen = True, visualize_dir = "visualize"):
    with tf.Session() as sess:
        if frozen:
            load_frozen_model(sess)
        else:
            build_graph(sess)
        summary_writer = tf.train.SummaryWriter(visualize_dir, sess.graph)
        summary_writer.flush()
        print "Graph written to '%s'. Use tensorboard to visualize." % visualize_dir
