# coding: utf-8

# In[1]:

#weather prediction
import pandas as pd
import numpy as np
#load image: reference: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
from skimage.io import imread_collection
pictures = np.array(imread_collection('katkam-scaled/*.jpg'))


# In[14]:

shape = pictures.shape
pictures


# In[15]:

X = pictures.reshape(shape[0],shape[1] * shape[2] * shape[3])


# In[13]:

print(X[1])


# In[ ]:


