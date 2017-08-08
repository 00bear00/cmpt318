
# coding: utf-8

# # <font color="red">CMPT318 PROJECT</font>
# MEMBER:
# - JEFF
# - KELVIN
# - JACKY

# In[ ]:

import pandas as pd
import numpy as np
from scipy import misc
from skimage.io import imread_collection
import re
import glob


# ## <font color="blue"> I. LOAD THE DATA </font>

# ### i. Load Webcam Images

# In[ ]:

image_collection = imread_collection('katkam-scaled/*.jpg')
images_np = np.array(image_collection)


# In[ ]:

# # IMAGE IN NUMPY ARRAY.
# images_np
# images_np.shape


# In[ ]:

images_df = pd.DataFrame(
    images_np.reshape(images_np.shape[0],images_np.shape[1] * images_np.shape[2] * images_np.shape[3]))


# In[ ]:

# IMAGE IN PANDAS DATAFRAME, AND BEEN RESHAPED.
# images_df


# ## ii. Load Weather Observations Data

# In[ ]:

#reference: for load multiple file from forder: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
weather_obv_filenames = glob.glob('yvr-weather/*.csv')
weather_obv_df = []

for filename in weather_obv_filenames:
    weather_obv_df.append(pd.read_csv(filename, skiprows=15)) # first 15 rows are general information, which is not useful data.

weather_obv_df = pd.concat(weather_obv_df).reset_index()


# In[ ]:

# WEATHER OBSERVATIONS DATAFRAME.
# weather_obv_df


# ## <font color="blue">II. CLEAN&PREPARE DATA</font>

# ### i. Clean Webcam Images

# In[ ]:

# GET IMAGE FILE_NAME
image_filenames = np.array(image_collection.files)


# In[ ]:

# extract image shoot date from filename
re_image_date = re.compile(r'katkam-\d\d\d\d\d\d\d\d\d\d\d\d\d\d')
def get_image_time(path):
    matches = re_image_date.findall(path)
    if matches:
        # preocess match learned from https://docs.python.org/2/library/re.html
        # take the last match which will be file name
        result = matches[-1]
        return (int(result[-14:-10]), int(result[-10:-8]), int(result[-8:-6]), result[-6:-4] + ':' + result[-4:-2])
    else:
        return 'wrong input file name format'
get_image_time = np.vectorize(get_image_time)
images_date = get_image_time(image_filenames)


# In[ ]:

# EXTRACTED IMAGE DATE
# images_date


# In[ ]:

# add relevant columns
images_df['Year'] = images_date[0]
images_df['Month'] = images_date[1]
images_df['Day'] = images_date[2]
images_df['Time'] = images_date[3]


# In[ ]:

# IMAGE DATA WITH DATE TIME
# images_df


# ## ii. Clean Weather Observations Data

# In[ ]:

# DROP UNNECESSARY COLUMNS
cleaned_weather_obv = weather_obv_df.drop(['index','Data Quality'], axis=1)
cleaned_weather_obv = cleaned_weather_obv.drop(['Temp Flag', 'Stn Press Flag','Wind Chill Flag', 'Hmdx Flag', 'Visibility Flag', 'Wind Spd Flag', 'Wind Dir Flag', 'Rel Hum Flag', 'Dew Point Temp Flag'], axis=1)


# In[ ]:

# SPLIT cleaned_weather INTO TWO DATAFRAME, ONE WITH NaN WEATHER COLUMN, ONE WITH NOT NaN COLUMN
data_whoseWeather_IsNaN = cleaned_weather_obv[~ cleaned_weather_obv.Weather.notnull()] # weather with Nan
main_training_data = cleaned_weather_obv[cleaned_weather_obv.Weather.notnull()] # weather without Nan


# In[ ]:

# TWO SPLIT DATAFRAME
# data_whoseWeather_IsNaN
# main_training_data.shape


# In[ ]:

main_training_data_withoutHW = main_training_data.drop(['Hmdx', 'Wind Chill'],axis=1)
final_data = main_training_data_withoutHW.dropna().copy()


# In[ ]:

# final_data


# ### weather description category include
# - Clear: Clear, Mainly Clear
# - Cloudy: Cloudy, Mostly Cloudy	
# - Fog: Fog, Freezing Fog,
# - Rain: Drizzle, Freezing Rain, Heavy Rain, Moderate Rain, Moderate Rain Showers, Rain, Rain Showers, Thunderstorms	
# - Snow: Moderate Snow, Snow Pellets	,Ice Pellets, Snow Showers
# ### we decide to use five categories :Clear, Cloudy, Fog, Rain, and Snow

# In[ ]:

# CHECK THE TOTAL VARIOUS WEATHER DESCRIPTION BEFORE CLEANING
# weather_category = final_data.groupby('Weather').count()
# weather_category


# In[ ]:

# for reduce # of class
def removeDuplicate(lst):
    lst = lst.split(",")
#     print(lst)
    newlst = ""
    for i in lst:
#         print (i)
        if i not in newlst:
            newlst = newlst + i + ','
    return newlst


# In[ ]:

re_clear = re.compile(r'Clear')
re_cloudy = re.compile(r'Cloudy')
re_fog = re.compile(r'Fog')
re_rain = re.compile(r'Rain')
re_snow = re.compile(r'Snow')
re_drizzle = re.compile(r'Drizzle')
re_thunderstorms = re.compile(r'Thunderstorms')
re_ice = re.compile(r'Ice')

# this output string
def clean_weather_description(Str):
    result = ''
    match_clear = re_clear.search(Str)
    if match_clear:
        result = result + match_clear[0]+','
        
    match_cloudy = re_cloudy.search(Str)
    if match_cloudy:
        result = result + match_cloudy[0]+','
        
    match_fog = re_fog.search(Str)
    if match_fog:
        result = result + match_fog[0]+','
        
    match_rain = re_rain.search(Str)
    if match_rain:
        result = result + match_rain[0]+','
        
    match_snow = re_snow.search(Str)
    if match_snow:
        result = result + match_snow[0]+','
        
    match_drizzle = re_drizzle.search(Str)
    if match_drizzle:
        result = result + 'Rain'+','
        
    match_thunderstorms = re_thunderstorms.search(Str)
    if match_thunderstorms:
        result = result + 'Rain'+','
        
    match_ice = re_ice.search(Str)
    if match_ice:
        result = result + 'Snow'+','
    
    result = removeDuplicate(result)
        
    return result[:-1]


# In[ ]:

# CLEAN THE WEATHER DESCRIPTION CATEGORY
final_data['Weather'] =final_data['Weather'].apply(clean_weather_description)


# In[ ]:

# final_data[final_data['Weather'] == 'Rain']
# SHOW THE NEW WEATHER CATEGORY
final_data
weather_category = final_data.groupby('Weather').count()
weather_category


# ## iii. Join Cleaned Webcam Image and Cleaned Weather obserbations Data Together

# In[ ]:

merged_data = final_data.merge(right = images_df, on = ['Year', 'Month', 'Day', 'Time'], how = 'inner')


# In[ ]:

merged_data


# In[ ]:

merged_data.iloc[:,13:]


# ## <font color="blue"> III. ANALYSE THE DATA </font>

# ### i. Use only image to predict weather description

#  ### 1. <font color="black">Split train_test data</font>

# In[ ]:

from sklearn.model_selection import train_test_split
X = merged_data.iloc[:,13:]
y = merged_data['Weather']
X_train, X_test, y_train, y_test = train_test_split(X, y)


# ### 2. <font color="black">try Bayesian Classifier</font>

# In[ ]:

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #within 1 min


# In[ ]:

from scipy import stats
print(stats.normaltest(X_train).pvalue)
# p-value too small,so no-normal
# don't use it


# ### 3. <font color="black">SVC</font>

# In[ ]:

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  #57 0.622103386809


# In[ ]:

# trying different parameter values
# model = SVC(kernel='rbf', decision_function_shape='ovr')     #score = 0.397504456328   around 23min
# model = SVC(kernel='linear')       # score = 0.668449197861   / 0.620320855615 ///0.641711229947
# model = SVC(kernel='rbf', decision_function_shape='ovr')   #score = 0.623885918004
model = SVC(kernel='linear', C=1e-1) #41   0.620320855615
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


# ### 3.1<font color="black">PCA + SVC</font>

# In[ ]:

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
model = make_pipeline(
    PCA(1000),
    SVC(kernel='linear', C=2.0)
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))   #0.614973262032/0.620320855615   37-39 2 min fast 18-20


# In[ ]:

# adjust C parameter
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
model = make_pipeline(
    PCA(1000),
    SVC(kernel='linear', C=0.1)
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) # little bit better 0.625668449198 / 0.6096256684492.run in 03min


# In[ ]:

# adjust PCA parameter
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
model = make_pipeline(
    PCA(100),
    SVC(kernel='linear', C=2.0)
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #02 - 36  very slow, why????? around 35 min


# In[ ]:

# adjust PCA parameter
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
model = make_pipeline(
    PCA(5000),
    SVC(kernel='linear', C=2.0)
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #with in 2 min  refine pca   0.654188948307


# ### 4. <font color="black">Nearest Neighbours</font>

# In[ ]:

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #4 min   0.609625668449 = neighbor5
#3.40 min   0.600713012478   meighbors=10         0.616755793226


# ### 5. <font color="black">neural_network</font>

# In[ ]:

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(), random_state=0)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))   # 1 min 0.393939393939


# In[ ]:

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 30),
                    random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) #  5 min 0.399286987522


# ## ii. Use image and weather conditions to predict weather description
# Jeff: I think if add weather condition into X, Scaler may need use. Because the unit is much different

# ### 1. add weather condition to X and split data set

# In[ ]:

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
X_condition = merged_data.drop(['Weather'], axis=1)
X_condition = X_condition.drop(['Time'], axis=1)
X_condition = X_condition.drop(['Date/Time','Year','Month','Day'], axis=1)
y_condition = merged_data['Weather']
X_condition_train, X_condition_test, y_condition_train, y_condition_test = train_test_split(X_condition, y_condition)


# ### 2. Bayesian classifier

# In[ ]:

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_condition_train, y_condition_train)
print(model.score(X_condition_test, y_condition_test)) #0.549019607843 less than 1 mins


# ### 3. Nearest Neighbours

# In[ ]:

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_condition_train, y_condition_train)
print(model.score(X_condition_test, y_condition_test)) #0.668449197861 n=10 5mins


# ### 4. <font color="black">SVC try add Scaler, not very different</font>

# In[ ]:

from sklearn.pipeline import make_pipeline   #with scaler
from sklearn.decomposition import PCA
model = make_pipeline(
    StandardScaler(),
    PCA(5000),
    SVC(kernel='linear', C=2.0)
)
model.fit(X_condition_train, y_condition_train)
print(model.score(X_condition_test, y_condition_test))  


# ### 4.1 SVC <font color="black">Without Scaler</font>

# In[ ]:

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
best_model = make_pipeline(
    PCA(5000),
    SVC(kernel='linear', C=2.0)
)
best_model.fit(X_condition_train, y_condition_train)
print(best_model.score(X_condition_test, y_condition_test))   
#0.704    this is the best accuracy we can get


# ## iii. <font color="black">try to deal with multilabel</font>
# 

# <font color="black">method 1): only use first weather in the weather column. eg: 'Rain,Fog' becomes 'Rain'. The result did not change too much.</font>

# In[ ]:

# def cleanMultiLabel(inputstr):
#     lst = inputstr.split(",")
#     if len(lst) > 1:
#         return lst[0]
#     return inputstr
# y_condition = merged_data['Weather'].apply(cleanMultiLabel)


# In[ ]:

# # y_condition   #there only one weather description in  'Weather' column.
# X_condition_train, X_condition_test, y_condition_train, y_condition_test = train_test_split(X_condition, y_condition)
# from sklearn.pipeline import make_pipeline
# from sklearn.decomposition import PCA
# model = make_pipeline(
#     PCA(5000),
#     SVC(kernel='linear', C=2.0)
# )
# model.fit(X_condition_train, y_condition_train)
# print(model.score(X_condition_test, y_condition_test))   
# #0.704  //////////////////////not very different to the one without clean multilable


# <font color="black"> method 2 :using multilabelbinarizer </font>

# In[ ]:

#change the type of Weather column
def change_str_to_array(s):
    return s.split(',')
# change y_train and y_test into proper shape
from sklearn.preprocessing import MultiLabelBinarizer
y_condition = MultiLabelBinarizer().fit_transform(merged_data['Weather'].apply(change_str_to_array))
X_condition_train, X_condition_test, y_condition_train, y_condition_test = train_test_split(X_condition, y_condition)
from sklearn.multiclass import OneVsRestClassifier
model = make_pipeline(
    PCA(5000),
    OneVsRestClassifier(SVC(kernel='linear', C=2.0))
)
model.fit(X_condition_train, y_condition_train)
print(model.score(X_condition_test, y_condition_test))  #57% with multilable, <<70.4. maybe because most row are single lablled.


# In[ ]:




# ## <font color="blue"> IV. PRESENT RESULT </font>

# ## <font color="black">using our best model,find what are the wrong predictions, and analy it to figure out why these predictions are wrong</font>

# In[ ]:


comparsion = pd.DataFrame({'truth': y_condition_test, 'prediction': best_model.predict(X_condition_test)})
result = merged_data.join(comparsion, how='outer')
result = result.dropna()
comparsion_with_date = result[['Date/Time','Time','prediction','truth']]
# comparsion_with_date


# 1) how many wrong prediction are there  for each hour of the day? 

# In[ ]:

difference = comparsion_with_date[comparsion_with_date['truth'] != comparsion_with_date['prediction']]
print(difference.groupby('Time').count())
print(merged_data.groupby('Time').count())
# no significant result. altough more wrong predictions in 7am ,10am,etc. The reason is # of points are different


# 2) beyond the above question, what is the percentages of wrong predictions in each hour of the day?

# In[ ]:

percentage_wrong_prediction = difference.groupby('Time').count() / comparsion_with_date.groupby('Time').count()
# highest percentage (most wrong predictions) for 6am maybe because before sun raise, hard to tell if its fog, rain, etc. low percentage (least wrong prediction) for 8 pm, 9 pm seems best but maybe we do not have enough data points at 8 pm and 9pm (because NA for weather column and we dropped it). 11am seems very good. 


# ## <font color="blue"> V. Other Interesting RESULT </font>

# ### i. Using image to predict time of the day, running the following code may cost about 10 minutes

# In[ ]:

#generate training set and test set
X = images_df.iloc[:,:-4]
y = images_df['Time']
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:

#trying to use SVC model 
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 0.46


# In[ ]:

#trying to use SVC model with PCA
model = make_pipeline(
    PCA(5000),
    SVC(kernel='linear', C=2.0)
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))   
# 0.5048. Better than Plain SVC.About half can be perdicated correctly. because things like 10am,11am probably looks very similar on the pictures, so it's hard to distinguish them 

