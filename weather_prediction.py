
import numpy as np
import pandas as pd
from skimage.io import imread_collection


#load image: reference: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
img_dir = 'katkam-scaled/*.jpg'
img_list = col = imread_collection(img_dir)