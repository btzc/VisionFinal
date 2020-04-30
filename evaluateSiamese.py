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
encoding_network.load_weights('./saved_model/encoding_network_weights.h5')

# 1) Calculate the 128 dimensional face encoding for each test image and calculate the distance
#    with all anchor images.
# 2) Save results in a csv file.

# Note that pd.read_csv function assumes that the first row is header and skips this row when reading..
reader = pd.read_csv('./test1.csv', chunksize=1)

# Load face encodings
#with open('anchor_encodings_dict.dat', 'rb') as f:
    #all_face_encodings = pickle.load(f)

write_header = True

for chunk in reader:
    encoding_net_test_inputs = np.empty((0, 224, 224, 3))
    test_img = image.load_img(chunk.iloc[0, 1], target_size=(224, 224))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = preprocess_input(test_img)
    encoding_net_test_inputs = np.append(encoding_net_test_inputs, test_img, axis=0)
    test_encoding = encoding_network.predict([encoding_net_test_inputs],
                                             batch_size = 1,
                                             verbose = 0)
    print(test_encoding)
    exit(0)

    #for (anchor_img_path, anchor_encoding) in all_face_encodings.items():
        #distance = np.linalg.norm(anchor_encoding - test_encoding)
        #chunk[anchor_img_path[-6:]] = distance  # only write the last 6 letters of each anchor path in the first row

    chunk.to_csv('test_set_prediction_results.csv', mode='a', header=write_header, index=False)
    write_header = False  # Update so later chunks don't write header

print("For each test image, the distance with each anchor image has been calculated and saved",
      "in the file test_set_prediction_results.csv")
