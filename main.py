import os
import cv2
import sys
import math
import numpy as np
import tensorflow as tf

from glob import glob
from random import randint
from utils import *

class DeepColour():

    def __init__(self, img_size=256, batch_size=4):
        self.batch_size = batch_size
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = img_size
        self.output_size = img_size

        self.df_dim = 64 # 64 disciminator filters
        self.gf_dim = 64

        self.input_bw = 1
        self.input_colours = 3 # Yes, 'coloUrs', OOOH CAAAANADAAAA!!
        self.output_colours = 3

        self.d_bn1 = batch_norm(name='d_bn_1')
        self.d_bn2 = batch_norm(name='d_bn_2')
        self.d_bn3 = batch_norm(name='d_bn_3')

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_bw])
        self.colour_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colours])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colours])

        # Generated images
        preimage = tf.concat([self.line_images, self.colour_images], 3)
        self.gen_images = self.generator(preimage)

        # Hold our "true" and "fake" data for disciminator
        # Basically [preimage (lines + colour hint), actual real image]
        #           [preimage, image generated from preimage]
        self.real_AB = tf.concat([preimage, self.real_images], 3)
        self.fake_AB = tf.concat([preimage, self.gen_images], 3)

        self.disc_true, disc_true_logits = self.discriminator(self.real_AB, reuse=False)
        self.disc_fake, disc_fake_logits = self.discriminator(self.fake_AB, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits)))\
                      + 100 * tf.reduce_mean(tf.abs(self.real_images - self.gen_images))

        # We'll need some way to make sure optimizers for d/g only edit d/g vars
        train_vars = tf.trainable_variables()
        self.d_vars = [var for var in train_vars if 'd_' in var.name]
        self.g_vars = [var for var in train_vars if 'g_' in var.name]

        with tf.variable_scope('discrim', reuse=tf.AUTO_REUSE):
            self.d_opt = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
            self.g_opt = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        # Just a basic conv with batch norm
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4), h4

    def generator(self, img_in):
        s = self.output_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        # Encoder part
        # Starting image is (if default) 256x256x4 (1c lineart, 3c colour hint)
        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv')          # 128 x 128 x 64
        e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # 64 x 64 x 128
        e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # 32 x 32 x 256
        e4 = bn(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # 16 x 16 x 512
        e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # 8 x 8 x 512

        # Decoder + Residual Connections
        # We work backwards now, from e5 -> e1, adding skip connections appropriately
        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
        d4 = bn(self.d4)
        d4 = tf.concat([d4, e4], 3)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
        d5 = bn(self.d5)
        d5 = tf.concat([d5, e3], 3)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = bn(self.d6)
        d6 = tf.concat([d6, e2], 3)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
        d7 = bn(self.d7)
        d7 = tf.concat([d7, e1], 3)

        # We're out of encoder layers! Now let's generate a proper image (hopefully) with colour!
        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colours], name='g_d8', with_w=True)
        # No need for batch_norm, we're just going to output this now
        # After much repetition, tanh function for output like DCGAN
        return tf.nn.tanh(self.d8)

    def train(self):
        self.load_model()

        data = glob(os.path.join('imgs', '*.jpg'))
        examples = np.array([get_image(img) for img in data[0:self.batch_size]])
        examples_normalized = examples / 255.0

        kernel = np.ones((2,2), np.uint8)
        # Is this a smart way to get edges? idk
        #examples_edge = np.array([cv2.erode(cv2.bitwise_not(cv2.Canny(img, 100, 100)), kernel, 2) for img in examples[0:self.batch_size]]) / 255.0
        examples_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ex, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ex in examples]) / 255.0
        examples_edge = np.expand_dims(examples_edge, 3)

        examples_colour = np.array([self.get_colourHints(img) for img in examples[0:self.batch_size]]) / 255.0

        ims('results/examples.jpg', merge_colour(examples_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims('results/examples_edges.jpg', merge(examples_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims('results/examples_colour_hints.jpg', merge_colour(examples_colour, [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(data)

        for e in range(10000):
            for i in range(int(datalen/self.batch_size)):
                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = np.array([get_image(img) for img in batch_files])
                batch_normed = batch / 255.0

                #batch_edges = np.array([cv2.erode(cv2.bitwise_not(cv2.Canny(img, 100, 100)), kernel, 2) for img in batch]) / 255.0
                batch_edges = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
                batch_edges = np.expand_dims(batch_edges, 3)
                batch_colour = np.array([self.get_colourHints(img) for img in batch]) / 255.0

                d_loss, _ = self.sess.run([self.d_loss, self.d_opt], feed_dict={self.line_images:batch_edges, self.colour_images:batch_colour, self.real_images:batch_normed})
                g_loss, _ = self.sess.run([self.g_loss, self.g_opt], feed_dict={self.line_images:batch_edges, self.colour_images:batch_colour, self.real_images:batch_normed})

                print('Epoch: %d  |  [%d / %d]  |  d_loss: %.4f  |  g_loss: %.4f' % (e, i, int(datalen/self.batch_size), d_loss, g_loss))

                if i % 10 == 0:
                    recreation = self.sess.run(self.gen_images, feed_dict={self.line_images:batch_edges, self.colour_images:batch_colour, self.real_images:batch_normed})
                    ims('results/' + str(e*10000 + i) + '.jpg', merge_colour(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

                if i % 30 == 0:
                    self.save('./checkpoint', e*10000 + i)

    def load(self, checkpoint_path):
        checkpoint_dir = os.path.join(checkpoint_path, 'tr')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def load_model(self, load_discim=True):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if load_discim:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)

        if self.load('./checkpoint'):
            print("Loaded from checkpoint")
        else:
            print("No checkpoint or error loading from checkpoint")

    def get_colourHints(self, img):
        for _ in range(30):
            random_x = randint(0, self.image_size-50)
            random_y = randint(0, self.image_size-50)

            img[random_x:random_x+50, random_y:random_y+50] = 255

        colour_hint = cv2.blur(img, (100,100))
        return colour_hint

    def save(self, checkpoint_dir, step):
        model_name = 'model'
        model_dir = 'tr'
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Use like so: python main.py [train]')
    else:
        cmd = sys.argv[1]
        if cmd == 'train':
            dc = DeepColour()
            dc.train()
