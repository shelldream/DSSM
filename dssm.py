# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date: 
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import os

import numpy as np
import tensorflow as tf
from common import xavier_init

class DSSM(object):
    def __init__(self, n_input, n_l1_hidden, n_l2_hidden, transfer_function=tf.nn.relu, \
                optimizer=tf.train.AdamOptimizer(), is_dropout=True, drop_rate=0.5, is_sparse=False, batch_size=1000):
        self.n_input = n_input
        self.n_l1_hidden = n_l1_hidden
        self.n_l2_hidden = n_l2_hidden
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.is_dropout = is_dropout
        self.drop_rate = drop_rate
        self.is_sparse = is_sparse
        self.batch_size = batch_size
        self.weight = self._initialize_weights()
        
        if self.is_sparse:
            self.query_batch = tf.sparse_placeholder(tf.float32, shape=np.array([batch_size, self.n_input], np.int64), name="QueryBatch")
            self.pos_doc_batch = tf.sparse_placeholder(tf.float32, shape=np.array([batch_size, self.n_input], np.int64), name="PosDocBatch")
            self.neg_doc_batch = tf.sparse_placeholder(tf.float32, shape=np.array([batch_size, self.n_input], np.int64), name="NegDocBatch")
            #self.query_batch = tf.sparse_placeholder(tf.float32, None, name="QueryBatch")
            #self.pos_doc_batch = tf.sparse_placeholder(tf.float32, None, name="PosDocBatch")
            #self.neg_doc_batch = tf.sparse_placeholder(tf.float32, None, name="NegDocBatch")
        else:
            self.query_batch = tf.placeholder(tf.float32, [None, self.n_input], name="QueryBatch")
            self.pos_doc_batch = tf.placeholder(tf.float32, [None, self.n_input], name="PosDocBatch")
            self.neg_doc_batch = tf.placeholder(tf.float32, [None, self.n_input], name="NegDocBatch")
        
        print self.query_batch.get_shape()
        print self.pos_doc_batch.get_shape()
        print self.neg_doc_batch.get_shape()

        # loss function
        self.query_output = self.get_model_output(self.query_batch)
        self.pos_doc_output = self.get_model_output(self.pos_doc_batch)
        self.neg_doc_output = self.get_model_output(self.neg_doc_batch)
        pos_sim = self._cal_cos_sim(self.query_output, self.pos_doc_output)
        neg_sim = self._cal_cos_sim(self.query_output, self.neg_doc_output)
        delta_sim = pos_sim - neg_sim
        self.loss = tf.reduce_mean(tf.sigmoid(delta_sim))
        self.optimizer = optimizer.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        #模型保存
        self.saver = tf.train.Saver()
        if not os.path.exists("./model"):
            os.popen("mkdir ./model")
        self.save_path = "./model/model_default.ckpt"

    def _cal_cos_sim(self, data1, data2):
        """计算两个维度相同的张量间的余弦距离"""
        data1_norm = tf.sqrt(tf.reduce_sum(tf.square(data1), 1, False)) 
        data2_norm = tf.sqrt(tf.reduce_sum(tf.square(data2), 1, False)) 
        norm_product = tf.multiply(data1_norm, data2_norm)
        product = tf.multiply(data1_norm, data2_norm)
        cos_sim = tf.truediv(product, norm_product)
        return cos_sim

    def get_model_output(self, data_batch):
        """构建模型""" 
        if self.is_sparse:
            data_l1 = tf.sparse_tensor_dense_matmul(data_batch, self.weight["w1"]) + self.weight["b1"]
        else:
            data_l1 = tf.matmul(data_batch, self.weight["w1"]) + self.weight["b1"]
        data_l1_output = self.transfer_function(data_l1)
        if self.is_dropout:
            data_l1_output = tf.nn.dropout(data_l1_output, self.drop_rate)
        
        data_l2 = tf.matmul(data_l1_output, self.weight["w2"]) + self.weight["b2"]
        data_l2_output = self.transfer_function(data_l2)
        if self.is_dropout:
            data_l2_output = tf.nn.dropout(data_l2_output, self.drop_rate)
        
        return data_l2_output 

    def _initialize_weights(self):
        """初始化模型参数"""
        all_weight = dict()
        all_weight["w1"] = tf.Variable(xavier_init(self.n_input, self.n_l1_hidden))
        all_weight["b1"] = tf.Variable(tf.zeros([self.n_l1_hidden], dtype=tf.float32))
        all_weight["w2"] = tf.Variable(xavier_init(self.n_l1_hidden, self.n_l2_hidden))
        all_weight["b2"] = tf.Variable(tf.zeros([self.n_l2_hidden], dtype=tf.float32))
        return all_weight

    def partial_fit(self, query_x, pos_doc_x, neg_doc_x):
        """用一个batch数据进行训练并返回当前的损失
            Args:
                X: 一个batch的训练数据
            Returns:
                loss: 当前训练的损失
        """
        loss, opt = self.sess.run((self.loss, self.optimizer), \
            feed_dict = {self.query_batch:query_x, self.pos_doc_batch:pos_doc_x, self.neg_doc_batch:neg_doc_x})
        return loss
    
    def save_model(self, save_path="./model/model_default.ckpt"):
        """保存模型至特定路径"""
        self.save_path = save_path
        self.saver.save(self.sess, save_path)
        print "The model has been saved in the file: %s"%self.save_path

    def load_model(self, save_path="./model/model_default.ckpt"):
        """载入已有模型"""
        if os.path.exists(save_path):
            self.saver.restore(save_path)
            print "The model %s has been loaded!"%save_path
