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


#load csv file
import glob

#reference: for load multiple file from forder: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
filenames = glob.glob('yvr-weather/*.csv')
all_df = []
for filename in filenames:
    all_df.append(pd.read_csv(filename, skiprows=15)) # first 15 rows are general information, which is not useful data.

weather_condition = pd.concat(all_df)

# pd.read_csv('weather-51442-201607.csv', skiprows=15)

