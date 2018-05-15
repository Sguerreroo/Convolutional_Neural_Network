# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 5


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)  # [float(np_float) for np_float in one_hot([i], num_classes)[0]]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=10, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


batch_train, label_batch_train = dataSource(["data/train/0/*.jpg", "data/train/1/*.jpg", "data/train/2/*.jpg"],
                                            batch_size=batch_size)
batch_valid, label_batch_valid = dataSource(["data/valid/0/*.jpg", "data/valid/1/*.jpg", "data/valid/2/*.jpg"],
                                            batch_size=batch_size)
batch_test, label_batch_test = dataSource(["data/test/0/*.jpg", "data/test/1/*.jpg", "data/test/2/*.jpg"],
                                          batch_size=batch_size)

batch_train_predicted = myModel(batch_train, reuse=False)
batch_valid_predicted = myModel(batch_valid, reuse=True)
batch_test_predicted = myModel(batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

def difference_in_error(errors):
    if len(errors) > 2:
        if abs(errors[-1] - errors[-2]) < 0.1 and errors[-1] < 0.05:
            return False
    return True


saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    epochs = 0
    train_errors = []
    valid_errors = []

    while epochs < 2000 and difference_in_error(valid_errors):
        sess.run(optimizer)
        if epochs % 20 == 0:
            train_error = sess.run(cost)
            train_errors.append(train_error)
            validation_error = sess.run(cost_valid)
            valid_errors.append(validation_error)
            print("Epoch:", epochs, "Train_error:", train_error, "Validation_error:", validation_error)
            print(sess.run(label_batch_valid))
            print(sess.run(batch_valid_predicted))
            print("---------------------------------------------")
        epochs += 1

    # NET ACCURACY
    mistakes = 0
    for _ in range(15):
        result = sess.run(batch_test_predicted)
        predicted = sess.run(label_batch_test)

        for b, r in zip(predicted, result):
            for i in range(3):
                if b[i] != round(r[i]):
                    mistakes += 1
                    break

    number_of_samples = len(sess.run(label_batch_test)) * 15
    print("Hit rate:", ((number_of_samples - mistakes) / number_of_samples) * 100, "%")

    import matplotlib.pyplot as plt

    plt.plot(train_errors)
    plt.plot(valid_errors)
    plt.legend(["Train Error", "Validation Error"])
    plt.show()

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
