import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

def compute_content_cost(a_C,a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(tf.transpose(a_C,[0,3,1,2]),[m,n_C,-1])
    a_G_unrolled = tf.reshape(tf.transpose(a_G,[0,3,1,2]),[m,n_C,-1])
    J_content = tf.reduce_sum(tf.squared_difference(a_C_unrolled,a_G_unrolled))/(4*n_H*n_W*n_C)
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A,A,transpose_b=True)
    return GA

def compute_layer_style_cost(a_S,a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(tf.transpose(a_S,[0,3,1,2]),[m,n_C,n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G,[0,3,1,2]),[m,n_C,n_H*n_W])
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = tf.reduce_sum(tf.squared_difference(GS,GG))/(4*(n_C**2)*((n_H*n_W)**2))
    return J_style_layer

def compute_style_cost(model, STYLE_LAYERS,sess):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J