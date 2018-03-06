#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import time
import tensorflow as tf

def weight_variable(shape):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

# No point in learning bias weights as they are cancelled
# out by the BatchNorm layers's mean adjustment.
def bn_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=False)

NCHW_enabled = False

def conv2d(x, W):
    if NCHW_enabled:
        return tf.nn.conv2d(x, W, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
    else:
        x = tf.transpose(x, [0, 2,3,1])
        t = tf.nn.conv2d(x, W,
                        strides=[1, 1, 1, 1], padding='SAME')
        t = tf.transpose(t, [0, 3, 1, 2])
        
        return t

class TFProcess:
    def __init__(self):
        # Network structure
        self.RESIDUAL_FILTERS = 128
        self.RESIDUAL_BLOCKS = 10

        # For exporting
        self.weights = []

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)


    def init(self, dataset, train_iterator, test_iterator):
        # TF variables
        print "init call"

    def init_net(self, weight, rs_blocks,rs_filters):
        self.x = tf.placeholder(tf.float32, [None, 18, 19 * 19])
        self.batch_norm_count = 0
        self.RESIDUAL_BLOCKS = rs_blocks
        self.RESIDUAL_FILTERS = rs_filters
        self.y_conv, self.z_conv = self.construct_net(self.x)


        self.avg_policy_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/test"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/train"), self.session.graph)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session.run(self.init)
        self.replace_weights(weight)

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            # Keyed batchnorm weights
            if isinstance(weights, str):
                work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                new_weight = tf.constant(new_weights[e])
                self.session.run(tf.assign(work_weights, new_weight))
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [2, 3, 1, 0])))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [1, 0])))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.session.run(weights.assign(new_weight))
        #This should result in identical file to the starting one
        #self.save_leelaz_weights('restored.txt')

    def restore(self, file):
        print("Restoring from {0}".format(file))
        self.saver.restore(self.session, file)

    def graph(self):
        return 

    def submit(self, planes):
        # We access the input and output nodes 
        x = self.session.graph.get_tensor_by_name('input_planes:0')                                                                                                  
        y = self.session.graph.get_tensor_by_name('policy_head:0')                                                                                                
        z = self.session.graph.get_tensor_by_name('value_head:0')
        y_out, z_out = self.session.run([y, z], feed_dict={x: planes})
        return (y_out,z_out)

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = weight_variable([filter_size, filter_size,
                                  input_channels, output_channels])
        b_conv = bn_bias_variable([output_channels])
        self.weights.append(W_conv)
        self.weights.append(b_conv)
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            h_bn = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv),
                    epsilon=1e-5, axis=1, fused=NCHW_enabled,
                    center=False, scale=False,
                    training=False)
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        orig = tf.identity(inputs)
        W_conv_1 = weight_variable([3, 3, channels, channels])
        b_conv_1 = bn_bias_variable([channels])
        self.weights.append(W_conv_1)
        self.weights.append(b_conv_1)
        weight_key_1 = self.get_batchnorm_key()
        self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

        # Second convnet
        W_conv_2 = weight_variable([3, 3, channels, channels])
        b_conv_2 = bn_bias_variable([channels])
        self.weights.append(W_conv_2)
        self.weights.append(b_conv_2)
        weight_key_2 = self.get_batchnorm_key()
        self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key_1):
            h_bn1 = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv_1),
                    epsilon=1e-5, axis=1, fused=NCHW_enabled,
                    center=False, scale=False,
                    training=False)
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2):
            h_bn2 = \
                tf.layers.batch_normalization(
                    conv2d(h_out_1, W_conv_2),
                    epsilon=1e-5, axis=1, fused=NCHW_enabled,
                    center=False, scale=False,
                    training=False)
        h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))
        return h_out_2

    def construct_net(self, planes):
        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19],  name='input_planes')

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=18,
                               output_channels=self.RESIDUAL_FILTERS)
        # Residual tower
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, self.RESIDUAL_FILTERS)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=2)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2*19*19])
        W_fc1 = weight_variable([2 * 19 * 19, (19 * 19) + 1])
        b_fc1 = bias_variable([(19 * 19) + 1])
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1,name="policy_head")

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=1)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19*19])
        W_fc2 = weight_variable([19 * 19, 256])
        b_fc2 = bias_variable([256])
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable([256, 1])
        b_fc3 = bias_variable([1])
        self.weights.append(W_fc3)
        self.weights.append(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3), name="value_head")

        return h_fc1, h_fc3
