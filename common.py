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
import numpy as np


def xavier_init(fan_in, fan_out, constant=1):
    """xavier 初始化器
       Args:
           fan_in: 输入节点个数
           fan_out: 输出节点个数
           constant: 常数系数
       Returns:
           res: tensor, 均匀分布,方差为特定值的tensor
    """
    low = -constant * np.sqrt(6.0 /(fan_in + fan_out))
    high = constant * np.sqrt(6.0 /(fan_in + fan_out))
    res = tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    return res


class DataBatchReader(object):
    """
        批量文件读取器
        读入的文件格式：每行三列
    """
    def __init__(self, dirname, field_cnt, record_default="", field_delim="\t", batch_size=5000):
        if not os.path.exists(dirname):
            raise ValueError("The data directory %s does not exist!"%dirname)
        self.dirname = dirname + "/"
        self.file_list = [self.dirname + filename for filename in os.listdir(self.dirname)]
        self.filename_queue = tf.train.string_input_producer(self.file_list, shuffle=True)
        self.batch_size = batch_size
        self.reader = tf.TextLineReader()
        key, value = self.reader.read(self.filename_queue)
        self.content = tf.decode_csv(value, record_defaults=[[record_default] for i in range(field_cnt)], field_delim=field_delim)
        self.features = tf.stack(self.content, axis=-1)

    def get_batch_data(self):
        min_after_dequeue = 10000
        capacity = min_after_dequeue +  self.batch_size
        batch_f = tf.train.shuffle_batch([self.features], batch_size=self.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(init)

            output = sess.run(batch_f)
            coord.request_stop()
            coord.join(threads)
        return output 

if __name__ == "__main__":
    test_dir = "/home/admin/huangxiaojun/data/DSSM"
    batch_reader = DataBatchReader(test_dir, field_cnt=3)
    output = batch_reader.get_batch_data()
