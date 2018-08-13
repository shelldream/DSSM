#-*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding('utf-8')
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


