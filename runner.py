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

dataset = load_train_test()
train = dataset['train']
test = dataset['test']

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

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

model = ResNet50(weights='imagenet')
print(model.layers.pop())


for layer in model.layers:
    layer.trainable = False

# All Batch Normalization layers still need to be trainable so that the "mean"
# and "standard deviation (std)" params can be updated with the new training data
model.get_layer('bn_conv1').trainable = True
model.get_layer('bn2a_branch2a').trainable = True
model.get_layer('bn2a_branch2b').trainable = True
model.get_layer('bn2a_branch2c').trainable = True
model.get_layer('bn2a_branch1').trainable = True
model.get_layer('bn2b_branch2a').trainable = True
model.get_layer('bn2b_branch2b').trainable = True
model.get_layer('bn2b_branch2c').trainable = True
model.get_layer('bn2c_branch2a').trainable = True
model.get_layer('bn2c_branch2b').trainable = True
model.get_layer('bn2c_branch2c').trainable = True
model.get_layer('bn3a_branch2a').trainable = True
model.get_layer('bn3a_branch2b').trainable = True
model.get_layer('bn3a_branch2c').trainable = True
model.get_layer('bn3a_branch1').trainable = True
model.get_layer('bn3b_branch2a').trainable = True
model.get_layer('bn3b_branch2b').trainable = True
model.get_layer('bn3b_branch2c').trainable = True
model.get_layer('bn3c_branch2a').trainable = True
model.get_layer('bn3c_branch2b').trainable = True
model.get_layer('bn3c_branch2c').trainable = True
model.get_layer('bn3d_branch2a').trainable = True
model.get_layer('bn3d_branch2b').trainable = True
model.get_layer('bn3d_branch2c').trainable = True
model.get_layer('bn4a_branch2a').trainable = True
model.get_layer('bn4a_branch2b').trainable = True
model.get_layer('bn4a_branch2c').trainable = True
model.get_layer('bn4a_branch1').trainable = True
model.get_layer('bn4b_branch2a').trainable = True
model.get_layer('bn4b_branch2b').trainable = True
model.get_layer('bn4b_branch2c').trainable = True
model.get_layer('bn4c_branch2a').trainable = True
model.get_layer('bn4c_branch2b').trainable = True
model.get_layer('bn4c_branch2c').trainable = True
model.get_layer('bn4d_branch2a').trainable = True
model.get_layer('bn4d_branch2b').trainable = True
model.get_layer('bn4d_branch2c').trainable = True
model.get_layer('bn4e_branch2a').trainable = True
model.get_layer('bn4e_branch2b').trainable = True
model.get_layer('bn4e_branch2c').trainable = True
model.get_layer('bn4f_branch2a').trainable = True
model.get_layer('bn4f_branch2b').trainable = True
model.get_layer('bn4f_branch2c').trainable = True
model.get_layer('bn5a_branch2a').trainable = True
model.get_layer('bn5a_branch2b').trainable = True
model.get_layer('bn5a_branch2c').trainable = True
model.get_layer('bn5a_branch1').trainable = True
model.get_layer('bn5b_branch2a').trainable = True
model.get_layer('bn5b_branch2b').trainable = True
model.get_layer('bn5b_branch2c').trainable = True
model.get_layer('bn5c_branch2a').trainable = True
model.get_layer('bn5c_branch2b').trainable = True
model.get_layer('bn5c_branch2c').trainable = True

x = model.get_layer('avg_pool').output

model_out = Dense(128, activation='relu',  name='model_out')(x)
model_out = Lambda(lambda  x: K.l2_normalize(x,axis=-1))(model_out)

new_model = Model(inputs=model.input, outputs=model_out)

anchor_input = Input(shape=(224, 224, 3), name='anchor_input')
pos_input = Input(shape=(224, 224, 3), name='pos_input')
neg_input = Input(shape=(224, 224, 3), name='neg_input')

encoding_anchor   = new_model(anchor_input)
encoding_pos      = new_model(pos_input)
encoding_neg      = new_model(neg_input)

loss = Lambda(triplet_loss)([encoding_anchor, encoding_pos, encoding_neg])

siamese_network = Model(inputs  = [anchor_input, pos_input, neg_input], outputs = loss)

siamese_network.compile(optimizer=Adam(lr=.00004, clipnorm=1.), loss=identity_loss)

train_path1 = './train1.csv'
TRAIN_INPUT_PATHS = [train_path1]

RECORD_DEFAULTS_TRAIN = [[0], [''], [''], ['']]

def decode_csv_train(line):
   parsed_line = tf.decode_csv(line, RECORD_DEFAULTS_TRAIN)
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

train_eval_score = 0

# Use the agnostic tensorflow session from Keras
sess = K.get_session()

#print("bn3d_branch2c: (Should be same)\n",
#      siamese_network.get_layer('model_1').get_layer('bn3d_branch2c').get_weights())
#print("res3d_branch2c: (Should be same)\n",
#      siamese_network.get_layer('model_1').get_layer('res3d_branch2c').get_weights())
print(siamese_network.get_layer('model_1').get_layer('bn3d_branch2c').trainable, "  Should be TRUE")
print(siamese_network.get_layer('model_1').get_layer('res3d_branch2c').trainable, "  Should be FALSE")

nr_epochs = 25
for i in range(0, nr_epochs):
    print("\nnr_epoch: ", str(i), "\n")
    sess.run(iterator.initializer, feed_dict={filenames: TRAIN_INPUT_PATHS})
    while True:
        try:
          anchor_path, pos_path, neg_path = sess.run(next_element)

          anchor_imgs = np.empty((0, 224, 224, 3))
          pos_imgs = np.empty((0, 224, 224, 3))
          neg_imgs = np.empty((0, 224, 224, 3))
          for j in range (0, len(anchor_path)):
              #print(anchor_path)
              anchor_img = image.load_img(anchor_path[j], target_size=(224, 224))
              anchor_img = image.img_to_array(anchor_img)
              #print(anchor_imgs.shape)
              anchor_img = np.expand_dims(anchor_img, axis=0)
              #print(anchor_imgs.shape)
              anchor_img = preprocess_input(anchor_img)
              anchor_imgs = np.append(anchor_imgs, anchor_img, axis=0)
              #print(anchor_img.shape)

              #print(test_path)
              pos_img = image.load_img(pos_path[j], target_size=(224, 224))
              pos_img = image.img_to_array(pos_img)
              pos_img = np.expand_dims(pos_img, axis=0)
              pos_img = preprocess_input(pos_img)
              pos_imgs = np.append(pos_imgs, pos_img, axis=0)
              #print(pos_img.shape)

              neg_img = image.load_img(neg_path[j], target_size=(224, 224))
              neg_img = image.img_to_array(neg_img)
              neg_img = np.expand_dims(neg_img, axis=0)
              neg_img = preprocess_input(neg_img)
              neg_imgs = np.append(neg_imgs, neg_img, axis=0)
              #print(neg_img.shape)

          #print("len(anchor_imgs): ", len(anchor_imgs))
          #print("pos_imgs[0].shape: ", pos_imgs[0].shape)
          #print("neg_imgs.shape: ", neg_imgs.shape)

          #print(labels)

          # dummy output, needed for being able to run the fit(..) function
          z = np.zeros(len(anchor_path))

          siamese_network.fit(x=[anchor_imgs, pos_imgs, neg_imgs],
                              y=z,
                              batch_size=train_batch_size,
                              epochs=1,
                              verbose=1,
                              callbacks=None,
                              validation_split=0.0,
                              validation_data=None,
                              shuffle=True,
                              class_weight=None,
                              sample_weight=None,
                              initial_epoch=0,
                              steps_per_epoch=None,
                              validation_steps=None)

        except tf.errors.OutOfRangeError:
          print("Out of range error triggered (looped through training set).")
          break

# Training completed at this point. Save the model architecture and weights.

# Save the Siamese Network architecture
siamese_model_json = siamese_network.to_json()
with open("saved_model/siamese_network_arch.json", "w") as json_file:
    json_file.write(siamese_model_json)
# save the Siamese Network model weights
siamese_network.save_weights('saved_model/siamese_model_weights.h5')

# create and save the Encoding Network to use in predictions later on
encoding_input = Input(shape=(224, 224, 3), name='encoding_input')
encoding_output   = new_model(encoding_input)
encoding_network = Model(inputs  = encoding_input,
                         outputs = encoding_output)

weights = siamese_network.get_layer('model_1').get_weights()
encoding_network.get_layer('model_1').set_weights(weights)

# Save the Encoding Network architecture
encoding_model_json = encoding_network.to_json()
with open("saved_model/encoding_network_arch.json", "w") as json_file:
    json_file.write(encoding_model_json)
# save the Encoding Network model weights
encoding_network.save_weights('saved_model/encoding_network_weights.h5')

