# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import os
import tensorflow as tf

class DataLoader(object):
    def __init__(self, sess, fileList, batchSize, numEpochs, numThreads, \
        reader=tf.TextLineReader, decoder=tf.decode_csv, delimiter=',', \
        shuffle=False, capacity=32):
        """
            Args:

            Returns:
                
        """
        self.sess = sess
        self.fileList = fileList
        self.fileQueue = tf.train.string_input_producer(self.fileList,\
         num_epochs=numEpochs, shuffle=shuffle, capacity=capacity)
        self.reader = reader
        self.delimiter = delimiter
        key, value = self.reader.read(self.fileQueue)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)