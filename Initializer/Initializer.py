#-*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
    Reference: https://zhuanlan.zhihu.com/p/25110150
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import tensorflow as tf
import numpy as np

class Initializer(object):
    def __init__(self, varName, weightsShape, initializer):
        self.varName = varName
        self.initializer = initializer
        self.weightsShape = weightsShape

    def init(self, mean=0.0, stddev=1.0):
        if self.initializer == 'truncated_normal':
            self.output = self.init_weights_truncated(mean, stddev)
        elif self.initializer == 'random_normal':
            self.output = self.init_weights_random(mean, stddev)
        elif self.initializer == 'xavier':
            self.output = self.init_weights_xavier()
        elif self.initializer == 'he_init':
            self.output = self.init_weights_he(mean, stddev)
        else:
            self.output = self.init_weights_random(mean, stddev)
        return self.output
    
    def getVarSummary(self):
        mean = tf.reduce_mean(self.output)
        tf.summary.scalar("mean", mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(self.output - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(self.output))
        tf.summary.scalar("min", tf.reduce_min(self.output))
        tf.summary.histogram("histogram", self.output)

    def init_weights_truncated(self, mean, stddev):
        return tf.get_variable(self.varName, shape=self.weightsShape,
          initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))

    def init_weights_random(self, mean, stddev):
        return tf.get_variable(self.varName, shape=self.weightsShape,
          initializer=tf.random_normal_initializer(mean=mean, stddev=stddev))

    def init_weights_xavier(self):
        return tf.get_variable(self.varName, shape=self.weightsShape,
          initializer=tf.contrib.layers.xavier_initializer())
    
    def init_weights_he(self, mean, stddev):
        return tf.get_variable(self.varName, shape=self.weightsShape,
            initializer=tf.random_normal_initializer(mean=mean, stddev=stddev))/tf.sqrt(self.weightsShape[0]/2.0)
