import pandas as pd
import numpy as np
import tensorflow as tf


def loadData(file_name):
    df = pd.read_csv(file_name, delim_whitespace=True)
    cols = list(df.columns.values)
    return df, cols

def makePlaceholders(N,df):
    m = df.values.shape[0]
    return tf.placeholder(tf.float32, shape=(m,N+1), name='x'), tf.placeholder(tf.float32, shape=(m,1), name='y'),m

def graphLinearReg(phX,phY):
    xtx = tf.matmul(tf.transpose(phX),phX)
    xtxi = tf.matrix_inverse(xtx)
    xty = tf.matmul(tf.transpose(phX),phY)
    return tf.matmul(xtxi,xty)

def addOnes(x,m):
    ones = ones = np.ones([m,1])
    return np.concatenate((x,ones),axis = 1)

def linearReg(file_name, colsX, colY):
    df, cols = loadData(file_name)
    N = len(colsX)
    phX,phY,m = makePlaceholders(N,df)
    graph = graphLinearReg(phX,phY)
    fd = {phX: addOnes(df[colsX].values,m), phY: df[colY].values}
    with tf.Session() as sess:
        output = sess.run(graph, feed_dict=fd)
        print(output)

linearReg("poverty.txt", ["Brth15to17","PovPct"], ["ViolCrime"])
