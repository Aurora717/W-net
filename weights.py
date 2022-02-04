#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np  
import math
import operator
import random
import time


# In[ ]:


#@tf.function
def gaussian_neighbor(shape_0, shape_1,radius, sigma_x):
    
    stime0 = time.time()
   
    X, Y = np.mgrid[:shape_0, :shape_1]# Getting x and y coordinades of all pixels in an image
    cord = np.vstack((X.ravel(), Y.ravel())).T # combining the x an y coordinates. 
    cord = cord.astype('float32') 
    # cord[1] is our center vertex, cord[1][0] is the x coordinate and the other is the y coordinate\
    
    
    #creating skeleton of graph 
    graph_np = np.zeros(shape = (shape_0*shape_1,shape_0*shape_1), dtype ='float16'  )
    print("graph shape ", graph_np.shape)

    a= []
    
    for i, (center_x ,center_y) in enumerate(cord): #iterating through all the pixels :(

            neighbors = (X - center_x)**2 + (Y - center_y)**2 #The norm of Xi - Xj, where center is the center pixel 

            #neighbors[neighbors > (radius - 1)] = 0 # assigning zeros to distances less than that of the radii 
            #neighbors[neighbors!=0] = np.exp((-1*neighbors[neighbors!=0])/np.square(sigma_x))# taking the gaussian for the rest
            graph = np.exp((-1*neighbors)/np.square(sigma_x))
            graph_np[i] =graph.ravel() #filling the graph 
    

    

    del(cord,X,Y) #Deleting unnecessary arrays for memory 
    
    graph_np = tf.sparse.from_dense(graph_np)
    graph_np = tf.dtypes.cast(graph_np, tf.float32)
    print("Runtime of obtaining gaussian Neighbor is ", (time.time() - stime0), " seconds")
    return graph_np
# In[6]:


# In[ ]:


def intensity(sigma_I, image_tensor): 
    
    sigma_I = tf.cast(sigma_I, 'float32')
    
    #Defining Skeleton
    #intensity_weight = np.zeros(shape = (image_tensor.shape[0],image_tensor.shape[1]*image_tensor.shape[2],image_tensor.shape[1]*image_tensor.shape[2] ), dtype ='float32'  )
    
   
        
        #Flattening the image
    flat_image= tf.reshape(image_tensor, shape = ( -1, image_tensor.shape[2]))
    flat_int = tf.reduce_sum(flat_image, axis = 1)/image_tensor.shape[2] #Taking the mean of the pixel value across all 3 channels i.e---> Shape = (x,y,z,3) ---> Shape = (x,y,z)
    cc = tf.matmul(flat_int[:,None], tf.ones_like(flat_int[None,:]))
    intensity_weight =tf.exp(-1*(tf.realdiv(tf.square((cc - tf.transpose(cc))),tf.square(sigma_I))))

    del(flat_image)
    del(flat_int)
    del(cc)
    
    #intensity_weight = tf.cast(intensity_weight, tf.float32)
    #print("Intensity Calculated")
    return intensity_weight
    


# In[ ]:


#@tf.function
def RAG(image_tensor,graph_Spatial, sigma_I=10):
    
    #graph_Spatial=gaussian_neighbor(radius, sigma_x, image_tensor)
    
    graph_Brightness = intensity(sigma_I, image_tensor)
    graph = tf.SparseTensor.__mul__(graph_Spatial, graph_Brightness)
    #print("RAG Calculated")
    return graph


# In[ ]:


#@tf.function
def normalized_cuts(image_tensor,prediction, classes, graph_spatial , sigma_I):
    #start1 = time.time()
    #weights = edge_weights(img_flatten, X_train[0].shape[0] , X_train[0].shape[1], std_intensity=3, std_position=1, radius=5)
    #weights = tf.sparse.from_dense(weights)
    K = classes

    loss = tf.constant(K, dtype = tf.float32)
    weights = RAG(image_tensor, graph_spatial, sigma_I )

    
    for A_k in range(K):
               

                
         
                prediction_unravel = tf.expand_dims((prediction[:,A_k]), axis=0)
           
                pred_matrix = tf.multiply(tf.transpose(prediction_unravel), prediction_unravel)
                #w = tf.cast(weights, 'float32')
                pred_matrix = tf.cast(pred_matrix, tf.float32)
                pcc = tf.SparseTensor.__mul__(weights,pred_matrix)
                #print(pcc)
                neo  = tf.sparse.reduce_sum(pcc)

                
                
                f = tf.ones(tf.shape(prediction[:,A_k]))
                f = tf.expand_dims( f, axis = 0)
                fc =tf.multiply(tf.transpose( prediction_unravel), f)
                fc = tf.cast(fc, tf.float32)
                
                fcc = tf.SparseTensor.__mul__(weights, tf.transpose(fc))

                deo = tf.sparse.reduce_sum(fcc)
               
           
                loss = loss - (neo/deo)
    
           
    #print("Total time taken for Loss Calculation is ", time.time()-start1, " seconds")
    return loss

    


# In[ ]:


def reconstruction_loss(MSE, prediction, X):
   
    #prediction = tf.reshape(prediction, (-1, prediction.shape[3]))
    #X  = tf.reshape(X, (-1, X.shape[3]))
    print("prediction type ", type(prediction))
    #recon_loss = tf.reduce_mean(tf.square(prediction-X))
    recon_loss = MSE(prediction, X)
    return recon_loss

