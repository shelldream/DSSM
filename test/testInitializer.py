#-*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding('utf-8')

sys.path.append("../")
import Initializer.Initializer as Init
import tensorflow as tf

if __name__ == "__main__":
	initializer = Init.Initializer("w", [128, 64], initializer="")
	w = initializer.init()
	print w