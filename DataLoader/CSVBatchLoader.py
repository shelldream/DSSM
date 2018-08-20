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

class CSVBatchLoader(object):
    def __init__(self, sess, fileList, fieldCnt, batchSize, numThreads, \
        delimiter=',', defaultRec="", shuffle=False, capacity=128):
        """
            CSV formatted batch data loader
            双队列，文件队列 + 样本生成队列
            Args:
                defaultRec: 字段缺失时的默认值

            Returns:
                
        """
        self.fileList = fileList
        for filename in self.fileList:
            if not os.path.exists(filename):
                raise ValueError("The data file %s does not exist!"%filename)
        self.sess = sess
        
        self.fileQueue = tf.train.string_input_producer(self.fileList, shuffle=shuffle, seed=10)
        key, value = tf.TextLineReader().read(self.fileQueue)
        self.content = tf.decode_csv(value, 
            record_defaults=[[defaultRec] for i in range(fieldCnt)], field_delim=delimiter)
        self.features = tf.stack(self.content, axis=-1)

        input_tensor_list = [self.features]
        self.batch_f = tf.train.batch(input_tensor_list, batch_size=batchSize, 
            capacity=capacity, num_threads=numThreads)
        self.coord = tf.train.Coordinator()  # 调度器
        self.threads = tf.train.start_queue_runners(coord=self.coord) # 开始将文件名填充到队列

    def get_one_batch(self):
        """获取一个batch的数据"""
        output = self.sess.run(self.batch_f)
        return output

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)