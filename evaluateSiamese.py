import pandas as pd
import tensorflow as tf
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
#import PIL
#from PIL import Image
from keras.models import Model
from keras.layers import Dense, Input, subtract, concatenate, Lambda, add, maximum
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.models import load_model, model_from_json
import numpy as np
import pickle
# load the encoding_network to make predictions based on trained network

json_file = open('./saved_model/encoding_network_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoding_network = model_from_json(loaded_model_json)
# load weights into new model
encoding_network.load_weights('./saved_model/encoding_network_weights.h5')
