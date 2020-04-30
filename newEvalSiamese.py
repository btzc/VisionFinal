import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Input, subtract, concatenate, Lambda, add, maximum
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.models import load_model, model_from_json
import numpy as np
from dataloader import load_train_test
from evaluation import *

def triplet_loss(inputs, dist='euclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 2 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))

    returned_loss = K.mean(loss)
    return returned_loss


json_file = open('./saved_model/encoding_network_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoding_network = model_from_json(loaded_model_json)
encoding_network.load_weights('./saved_model/encoding_network_weights.h5')

test_path1 = './test1.csv'
TEST_INPUT_PATHS = [test_path1]

RECORD_DEFAULTS_TEST = [[0], [''], [''], ['']]

def decode_csv_train(line):
   parsed_line = tf.decode_csv(line, RECORD_DEFAULTS_TEST)
   anchor_path = parsed_line[1]
   pos_path  = parsed_line[2]
   neg_path    = parsed_line[3]
   return anchor_path, pos_path, neg_path

iterator_batch_size = 64
train_batch_size = 8

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(1).map(decode_csv_train))
dataset = dataset.shuffle(buffer_size=100000)
dataset = dataset.batch(iterator_batch_size)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
# Use the agnostic tensorflow session from Keras
sess = K.get_session()
sess.run(iterator.initializer, feed_dict={filenames: TEST_INPUT_PATHS})
losses = []
while True:
    try:
        anchor_path, pos_path, neg_path = sess.run(next_element)

        anchor_imgs = np.empty((0, 224, 224, 3))
        pos_imgs = np.empty((0, 224, 224, 3))
        neg_imgs = np.empty((0, 224, 224, 3))
        encoding_net_test_inputs = np.empty((0, 224, 224, 3))
        for j in range (0, len(anchor_path)):
            anchor_img = image.load_img(anchor_path[j], target_size=(224, 224))
            anchor_img = image.img_to_array(anchor_img)
            anchor_img = np.expand_dims(anchor_img, axis=0)
            anchor_img = preprocess_input(anchor_img)
            anc_encoding_net_test_inputs = np.append(encoding_net_test_inputs,
                                                anchor_img, axis=0)
            anc_encoding = encoding_network.predict([anc_encoding_net_test_inputs],
                                                batch_size = 1,verbose = 1)

            pos_img = image.load_img(pos_path[j], target_size=(224, 224))
            pos_img = image.img_to_array(pos_img)
            pos_img = np.expand_dims(pos_img, axis=0)
            pos_img = preprocess_input(pos_img)
            pos_encoding_net_test_inputs = np.append(encoding_net_test_inputs,
                                                pos_img, axis=0)
            pos_encoding = encoding_network.predict([pos_encoding_net_test_inputs],
                                                batch_size = 1,verbose = 1)

            neg_img = image.load_img(neg_path[j], target_size=(224, 224))
            neg_img = image.img_to_array(neg_img)
            neg_img = np.expand_dims(neg_img, axis=0)
            neg_img = preprocess_input(neg_img)
            neg_encoding_net_test_inputs = np.append(encoding_net_test_inputs,
                                                neg_img, axis=0)
            neg_encoding = encoding_network.predict([neg_encoding_net_test_inputs],
                                                batch_size = 1,verbose = 1)
            loss = Lambda(triplet_loss)([anc_encoding, pos_encoding, neg_encoding])
            print("This is loss: ", loss)
            losses.append(loss)

    except tf.errors.OutOfRangeError:
        print("Out of range error triggered (looped through training set).")
        break

print(np.mean(losses))
