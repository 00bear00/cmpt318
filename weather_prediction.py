
# coding: utf-8

# In[1]:

#weather prediction
import pandas as pd
import numpy as np
from scipy import misc
from skimage.io import imread_collection
import re


# In[2]:

image_collection = imread_collection('katkam-scaled/*.jpg')
images = np.array(image_collection)
shape = images.shape
# pictures


# In[3]:

images_df = pd.DataFrame(images.reshape(shape[0],shape[1] * shape[2] * shape[3]))


# In[4]:

images_df


# In[5]:

filenames = np.array(image_collection.files)
print(filenames[0])


# In[21]:

expression = re.compile(r'katkam-\d\d\d\d\d\d\d\d\d\d\d\d\d\d')
def get_time(path):
    matches = expression.findall(path)
    if matches:
        # preocess match learned from https://docs.python.org/2/library/re.html
        # take the last match which will be file name
        result = matches[-1]
        return (int(result[-14:-10]), int(result[-10:-8]), int(result[-8:-6]), result[-6:-4] + ':' + result[-4:-2]) 
    else:
        return "wrong input file name format"
get_time = np.vectorize(get_time)
times = get_time(filenames)


# In[24]:

images_df['Year'] = times[0]
images_df['Month'] = times[1]
images_df['Day'] = times[2]
images_df['Time'] = times[3]


# In[25]:

images_df


# In[16]:




# In[ ]:



