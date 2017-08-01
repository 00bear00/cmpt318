
#weather prediction
import pandas as pd
import numpy as np
from scipy import misc
from skimage.io import imread_collection
import re




image_collection = imread_collection('katkam-scaled/*.jpg')
images = np.array(image_collection)
shape = images.shape
# pictures



images_df = pd.DataFrame(images.reshape(shape[0],shape[1] * shape[2] * shape[3]))


images_df



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




#load csv file
import glob

#reference: for load multiple file from forder: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
filenames = glob.glob('yvr-weather/*.csv')
all_df = []
for filename in filenames:
    all_df.append(pd.read_csv(filename, skiprows=15)) # first 15 rows are general information, which is not useful data.

weather_condition = pd.concat(all_df)

# pd.read_csv('weather-51442-201607.csv', skiprows=15)






#cleaning data
cleaned_data = weather_condition.drop(['index','Data Quality'], axis=1)
cleaned_data = cleaned_data.drop(['Temp Flag', 'Stn Press Flag','Wind Chill Flag', 'Hmdx Flag', 'Visibility Flag', 'Wind Spd Flag', 'Wind Dir Flag', 'Rel Hum Flag', 'Dew Point Temp Flag'], axis=1)

#type(weather_condition['Year'][0])

data_whoseWeather_IsNaN = cleaned_data[~ weather_condition.Weather.notnull()] # weather with Nan
main_training_data = cleaned_data[weather_condition.Weather.notnull()] # weather without Nan

main_training_data_withoutHW = main_training_data.drop(['Hmdx', 'Wind Chill'],axis=1)
final_data = main_training_data_withoutHW.dropna()
# main_training_data = cleaned_data.copy()

#data_whoseWeather_IsNaN
#main_training_data







# Jacky
result = final_data.groupby(by=['Weather']).count()
import re
rain = re.compile(r'Rain')
snow = re.compile(r'Snow')
def simpleWeather(inputStr):
    matchesRain = rain.search(inputStr)
    result = ''
#     print(matches[0])
    if matches:
        result = result + matches[0]
    return result

simpleWeather('Showers,Snow Showers')
