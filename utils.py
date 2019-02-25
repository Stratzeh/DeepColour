import cv2
import numpy as np
import tensorflow as tf

class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)


batch_norm_count = 0

def batch_norm_reset():
    global batch_norm_count
    batch_norm_count = 0

def bn(x):
    global batch_norm_count
    batch_object = batch_norm(name='bn' + str(batch_norm_count))
    batch_norm_count += 1
    return batch_object(x)

def conv2d(input, output_dim, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, [1,s_h,s_w,1], 'SAME')

        b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv

def deconv2d(input, output_shape, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=[1,s_h,s_w,1])
        b = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        if with_w:
            return deconv, w, b

        return deconv

def linear(input, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input, matrix) + b, matrix, b

        return tf.matmul(input, matrix) + b

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def imread(path):
    return cv2.imread(path, 1)

def transform(img, crop_size=256):
    crop = cv2.resize(img, (crop_size, crop_size))
    return np.array(crop)

def get_image(path):
    return transform(imread(path))

def merge(images, size):
    h, w = images.shape[1:3]
    merged_img = np.zeros((size[0]*h, size[1]*w, 1))

    for idx, img in enumerate(images):
        i = int(idx / size[1])
        j = int(idx % size[1])
        merged_img[i*h:i*h+h, j*w:j*w+w] = img

    return merged_img[:,:,0]

def merge_colour(images, size):
    h, w = images.shape[1], images.shape[2]
    merged_img = np.zeros((size[0]*h, size[1]*w, 3))

    for idx, img in enumerate(images):
        i = int(idx / size[1])
        j = int(idx % size[1])
        merged_img[i*h:i*h+h, j*w:j*w+w] = img

    return merged_img

def ims(name, img):
    print('Saving image:', name)
    cv2.imwrite(name, img*255) # We want the actual pixel values, not the normalized ones
