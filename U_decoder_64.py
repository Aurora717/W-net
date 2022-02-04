#!/usr/bin/env python
# coding: utf-8

# In[1]:


from modules import conv_block, separable_conv_block,conv_transpose, out_sep_conv_block
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, SeparableConv2D, Conv2DTranspose,concatenate
from tensorflow.keras.layers import MaxPooling2D
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from IPython.display import  display
from PIL import Image
import random
import cv2
import time 
from tensorflow.keras import backend as K
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


class U_Decoder(keras.Model):
    def __init__(self, num_channels, activation = 'linear'):
        super(U_Decoder, self).__init__()
       # self.inputs = inputs
        
        self.conv1 =  conv_block(16, 0.1)
        self.conv2 =  separable_conv_block(32, 0.1)
        #self.conv3 =  separable_conv_block(64, 0.1)
        #self.conv4 =  separable_conv_block(128, 0.2)
        
        self.out_conv =  out_sep_conv_block(64, 0.2)
        
        #self.up_conv1 = conv_transpose(128,0.2)
        #self.up_conv2 = conv_transpose(64,0.1)
        self.up_conv3 = conv_transpose(32,0.1)
        self.up_conv4 = conv_transpose(16,0.1)
        
        #self.out_layer = Conv2D(num_classes,(1,1), activation = 'softmax'), Wnet version 
        
        self.out_layer = Conv2D( num_channels, (1,1), activation = activation) 
       
      
        
        
    def call(self, inputs):
        

        
        con1,pol1 = self.conv1(inputs)
         
   
        con2,pol2 = self.conv2(pol1)
       
        #con3,pol3 = self.conv3(pol2)
     
        #con4,pol4 = self.conv4(pol3)
       
        con5 = self.out_conv(pol2)

        #con6 = self.up_conv1(con5, con4)

        #con7 = self.up_conv2(con5, con3)

        con8 = self.up_conv3(con5, con2)

        
        con9 = self.up_conv4(con8, con1)
       
        out = self.out_layer(con9)
    

        return out
        
        
        


# In[ ]:




