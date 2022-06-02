import time
import numpy as np
# import _pickle as cPickle
import tensorflow as tf
from config import *
import random

ORIENTATIONS = ['L','R','U','D']

def get_time():
    return time.strftime("%a_%d_%b_%Y", time.localtime())

def save_pkl(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print("  [*] save %s" % path)

def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
        print("  [*] load %s" % path)
        return obj

def clipped_error(x):
    # Huber loss
    try:
        return tf.where(tf.abs(x) > 1.0, tf.abs(x) - 0.5, 0.5 * tf.square(x))
    except:
        return tf.where(tf.abs(x) > 1.0, tf.abs(x) - 0.5, 0.5 * tf.square(x))

def conv2d(inputs,
           kernel_size,
           channels,
           output_dim,
           strides,
           # c_names,
           initializer,
           activation_fn,
           data_format=data_format,
           padding='VALID',
           name='conv2d'):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, strides, strides]
            kernel_shape = [kernel_size, kernel_size, channels, output_dim]
        elif data_format == 'NHWC':
            stride = [1, strides, strides, 1]
            kernel_shape = [kernel_size, kernel_size, inputs.get_shape()[-1], output_dim]

        # w = tf.get_variable(name=name + '-weights', shape=[kernel_size, kernel_size, channels, output_dim], initializer=initializer)
        # conv = tf.nn.conv2d(inputs, w, strides=[1, 1, strides, strides], padding="VALID")
        w = tf.get_variable(name=name + '-weights', shape=kernel_shape, initializer=initializer)
        conv = tf.nn.conv2d(inputs, w, strides=stride, padding="VALID")

        # b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # b = tf.get_variable(name=name + '-biases', shape=[output_dim], initializer=tf.constant_initializer(0.0), collections=c_names)
        b = tf.get_variable(name=name + '-biases', shape=[output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)
        out = activation_fn(out)

    return out, w, b

def conv2dv2(x,
           output_dim,
           kernel_size,
           stride,
           c_names,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           padding='VALID',
           data_format=data_format,
           name='conv2d'):
    with tf.variable_scope(name):
        # stride = [1, 1, stride[0], stride[1]]
        # kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]

        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        # w = tf.get_variable(name + '-weights', kernel_shape, tf.float32, initializer=initializer, collections=c_names)
        w = tf.get_variable(name + '-weights', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format = data_format)

        # b = tf.get_variable(name + '-biases', [output_dim], initializer=tf.constant_initializer(0.0), collections=c_names)
        b = tf.get_variable(name + '-biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)
        out = activation_fn(out)

    return out, w, b


# def linear(inputs, hiddens, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
def linear(inputs, hiddens, c_names, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable(name=name + '-weights', shape=[shape[1], hiddens], initializer = tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable(name=name + '-biases', shape=[hiddens], initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(inputs, w), b)

        if activation_fn != None:
            return activation_fn(out), w, b
        else:
            return out, w, b

def get_action_with_target(y, x, direction, target):
    '''Get corresponding action according to agent position and orientation'''
    if target.x - x > 0:
        if direction == 'U':
            return 'MR'
        elif direction == 'D':
            return 'ML'
        elif direction == 'L':
            return 'MB'
        elif direction == 'R':
            return 'MF'
    elif target.x - x < 0:
        if direction == 'U':
            return 'ML'
        elif direction == 'D':
            return 'MR'
        elif direction == 'R':
            return 'MB'
        elif direction == 'L':
            return 'MF'
    elif target.y - y > 0:
        if direction == 'U':
            return 'MB'
        elif direction == 'D':
            return 'MF'
        elif direction == 'L':
            return 'ML'
        elif direction == 'R':
            return 'MR'
    elif target.y - y < 0:
        if direction == 'U':
            return 'MF'
        elif direction == 'D':
            return 'MB'
        elif direction == 'L':
            return 'MR'
        elif direction == 'R':
            return 'ML'
    else:
        return 'W'

def legal_moves():
    '''Returns a list of legal moves'''
    # step forward, step backward, step left, step right, rotate left, rotate right, use beam and stand still

    legal_moves = set(['W'])
    legal_moves.add('MF')
    legal_moves.add('MB')
    legal_moves.add('ML')
    legal_moves.add('MR')
    legal_moves.add('RL')
    legal_moves.add('RR')
    legal_moves.add('B')
    return legal_moves

def spawn_rotation():
    """Return a randomly selected initial rotation for an agent"""
    return random.choice(ORIENTATIONS)
