# -*- coding: utf-8 -*-

# Sample code to use string producer.

import numpy as np
import tensorflow as tf

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

n_input = 80 * 140


x = tf.placeholder(tf.float32, [None, 80, 140, 1])
with tf.variable_scope('ConvNet', reuse=False):
    o1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, activation=tf.nn.relu)
    o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
    o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
    o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

    h = tf.layers.dense(inputs=tf.reshape(o4, [1, 18*33*64]), units=10, activation=tf.nn.relu)
    y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)

# --------------------------------------------------
#
#       PLAY
#
# --------------------------------------------------

import cv2

cap = cv2.VideoCapture(0)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    while True:
        ret, img = cap.read()  # 720x1280x3 <-- print(img.shape);

        resized = cv2.resize(img, (140, 80), interpolation=cv2.INTER_AREA)
        #cropped = resized[0:180, 70:250]
        #resized64 = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
        gray = np.asarray(cv2.cvtColor(resized, 7))#PONER 7

        cv2.imshow('Capture', gray)
        frame = gray.reshape(-1, 80, 140, 1)
        print( sess.run(y, feed_dict={x: frame}))

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

cv2.destroyAllWindows()