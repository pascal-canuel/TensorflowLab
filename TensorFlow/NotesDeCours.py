#########################################################
#
#   Exemple dans les notes de cours
#   
#   https://cours.dinf.cll.qc.ca/sysindustriel/IA/IA
#
#########################################################

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import cv2
import time
import matplotlib.pyplot as plt

#########################################################
# 
#   https://cours.dinf.cll.qc.ca/sysindustriel/IA/TensorFlowBase/#21-graphes
#
#########################################################
def BaseGraphe(XData,YData):

    # Model input and output
    MonGraphe=tf.Graph()
    with MonGraphe.as_default():
        x=tf.Variable(XData,name="x")
        y= tf.Variable(YData,name="y")
        f= y*x+2*x-6

    #MonGraphe.x=tf.Variable(XData,name="x")
    #MonGraphe.y= tf.Variable(YData,name="y")
    #MonGraphe.f= MonGraphe.y* MonGraphe.x+ 2* MonGraphe.x-6

    #x= tf.Variable(XData,name="x")
    #y= tf.Variable(YData,name="y")
    #f = y*x+2*x-6 

    # Create interractive session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        init.run()
        result=f.eval()
        print("Graphe par defaut:",result)
        #result=MonGraphe.f.eval()
        #print("MonGraphe:",result)
        sess.close()
    return
#########################################################
# 
#   https://cours.dinf.cll.qc.ca/sysindustriel/IA/TensorFlowBase/#23-evaluation-des-noeuds
#
#########################################################
def MultiNode():
    w = tf.constant(3)
    x=w+2
    y=x+5
    z=x*3
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    init.run()
    #result=y.eval()
    #print("Valeur Y:",result)
    #result=z.eval()
    #print("Valeur Z:",result)

    resulty, resultz = sess.run([y,z])
    print("Valeur Y:",resulty)
    print("Valeur Z:",resultz)
    sess.close()
    return

#########################################################
# 
#  https://cours.dinf.cll.qc.ca/sysindustriel/IA/TensorFlowBase/#22-matrice
#
#########################################################
def Matrice():
    TabA = tf.constant([[1,2,3,4,5,],[-1,-2,-3,-4,-5]])
    TabB = tf.constant([[100,100,100,100,100],[200,200,200,200,200]])
    F1=TabA*TabB;
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    init.run()
    result=F1.eval()
    print("F1 result:",result)
    sess.close()
    return

#########################################################
# 
#  https://cours.dinf.cll.qc.ca/sysindustriel/IA/TensorFlowBase/#24-type-de-donnees
#
#########################################################
def Variable():
    Var1 = tf.Variable(3)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    init.run()
    print("Var1 step 1:",Var1.eval())
    Var1=Var1+2
    print("Var1 step 2:",Var1.eval())
    sess.close()
    
    sess = tf.InteractiveSession()
    init.run()
    print("Var1 step 3:",Var1.eval())
    Var1=Var1+2
    print("Var1 step 4:",Var1.eval())
    sess.close()
    return

#########################################################
# 
#  https://cours.dinf.cll.qc.ca/sysindustriel/IA/TensorFlowBase/#24-type-de-donnees
#
#########################################################
def EspaceReserve():
    x = tf.placeholder(tf.float32, shape=(1024, 1024))
    y = tf.matmul(x, x)

    with tf.Session() as sess:
        #print(sess.run(y))  # ERROR: will fail because x was not fed.
        rand_array = np.random.rand(1024, 1024)
        print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
    return