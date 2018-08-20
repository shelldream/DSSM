#-*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding('utf-8')

sys.path.append("../")
import DataLoader.CSVBatchLoader
import tensorflow as tf

def CSVLoaderTest():
	fileList = ["testData/data1.txt", "testData/data2.txt",
		 "testData/data3.txt", "testData/data4.txt"]
	fieldCnt = 3
	batchSize = 7
	numThreads = 1
	with tf.Session() as sess:
		dataLoader = DataLoader.CSVBatchLoader.CSVBatchLoader(sess, 
		fileList, fieldCnt, batchSize, numThreads, shuffle=True)
		for i in range(10):
			print dataLoader.get_one_batch()
			print "#"* 60
		dataLoader.close()

if __name__ == "__main__":
	CSVLoaderTest()