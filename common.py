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
        随机批量文件读取器
        读入的文件格式
    """
    def __init__(self, sess, dirname, field_cnt=3, record_default="", field_delim="\t", batch_size=5000):
        """
            Args:
                dirname: str, 数据存储的路径
                field_cnt: int, 数据文件每行的字段数
                field_delim: str, 每个字段的分隔符
                batch_size: int, 每个batch数据大小
            Returns:
                None
        """
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
        
        min_after_dequeue = 10000
        capacity = min_after_dequeue +  self.batch_size
        self.batch_f = tf.train.shuffle_batch([self.features], batch_size=self.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        self.sess = sess
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def get_one_batch(self):
        """获取一个batch的数据"""
        output = self.sess.run(self.batch_f)
        return output 

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
