#-*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import tensorflow as tf

class LossDefiner(object):
	def __init__(self, lossType):
		pass
	def triplet_hinge_loss(self, anchor_embedding, positive_embedding, negative_embedding, margin=0):
	    anchor_pos_loss = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding),1)
	    anchor_neg_loss = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding),1)
	    hingeloss = tf.maximum(0.0, margin + anchor_pos_loss - anchor_neg_loss)
	    hinge_loss = tf.reduce_mean(hingeloss)
	    return hinge_loss
	    
    def _cal_cos_sim(self, data1, data2):
	    """计算两个维度相同的张量间的余弦距离"""
	    data1_norm = tf.sqrt(tf.reduce_sum(tf.square(data1), 1, False)) 
	    data2_norm = tf.sqrt(tf.reduce_sum(tf.square(data2), 1, False)) 
	    norm_product = tf.multiply(data1_norm, data2_norm)
	    product = tf.multiply(data1_norm, data2_norm)
	    cos_sim = tf.truediv(product, norm_product)
	    return cos_sim