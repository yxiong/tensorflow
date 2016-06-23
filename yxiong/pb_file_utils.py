#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Jun 23, 2016.

import tensorflow as tf
from tensorflow.python.client import graph_util


"""Some utilities for manipulating protobuf model files."""


def extract_sub_graph(input_pb_file, output_nodes, output_pb_file):
    """Given an `input_pb_file` of a graph, extract the subgraph that can reach the `output_nodes`,
    trimming nodes that cannot. The trimmed graph will be written to `output_pb_file`."""
    graph_def = tf.Graph().as_graph_def()
    with open(input_pb_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    if isinstance(output_nodes, basestring):
        output_nodes = [output_nodes]
    sub_graph_def = graph_util.extract_sub_graph(graph_def, output_nodes)
    tf.train.write_graph(sub_graph_def, "", output_pb_file, as_text = False)


def rename_output_nodes(input_pb_file, name_change_map, output_pb_file):
    with tf.Session() as sess:
        graph_def = sess.graph.as_graph_def()
        with open(input_pb_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name = "")
        for old_name, new_name in name_change_map.iteritems():
            tensor = sess.graph.get_tensor_by_name(old_name + ":0")
            tf.identity(tensor, name = new_name)
        tf.train.write_graph(sess.graph.as_graph_def(), "", output_pb_file, as_text = False)


def visualize_graph(pb_file, visualize_dir):
    """Create visualization data of the graph in `pb_file` to `visualize_dir`. The output can be
    then visualized with `tensorboard`."""
    with tf.Session() as sess:
        graph_def = sess.graph.as_graph_def()
        with open(pb_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name = "")
        summary_writer = tf.train.SummaryWriter(visualize_dir, sess.graph)
        summary_writer.flush()
