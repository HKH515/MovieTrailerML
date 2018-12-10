from collections import defaultdict
import numpy as np
import pandas as pd
from keras.optimizers import SGD, rmsprop, Adam, Nadam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import cv2
import csv
import json

def longest_trailer(frame_info):
    max_v = 0
    max_k = None
    for movieId in os.listdir("frames"):
        print(os.listdir("frames/%s" % movieId))
        candidate_v = len(os.listdir("frames/%s" % movieId))
        if candidate_v > max_v and movieId in frame_info['good']:
            max_v = candidate_v
            max_k = movieId
    return max_k, max_v

class data_generator(Sequence):
    def __init__(self, ids, labels, batch_size, z_num=10):
        self.ids, self.labels = ids, labels
        self.batch_size = batch_size
        self.z_num = z_num
    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))
    def __getitem__(self, idx):
        batch_x = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = [] 
        for movieId in batch_x:
            frames = []
            #print("Dealing with movieId {}".format(movieId))
            for frame in os.listdir("frames/%s" % movieId):
                frame_data = cv2.imread("frames/%s/%s" % (movieId, frame))
                frame_data = cv2.resize(frame_data, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
                frames.append(frame_data)
                if len(frames) >= self.z_num:
                    break
            

            # Pad zeros to the end s.t. the length is the max of the whole set
            frames.extend([np.zeros(frame_data.shape) for _ in range(self.z_num - len(frames))])
            batch_data.append(frames)
        return np.array(batch_data), np.array(batch_y)

with open('frame_info.json', 'r') as f:
    frame_info = json.load(f)

# get the data
df = pd.read_csv('preprocessed_movies.csv')
prune_mask = df['movieId'].isin(frame_info['good'])
df = df.loc[prune_mask]

genres = df.columns[4:].tolist()
frame_sequences = []
labels = []

# drop all data not needed for machine learning
data = df.loc[:,'movieId']
labels = df.loc[:, genres].values.tolist()

# find longest trailer
longest_trailer_movieId, longest_trailer_length = longest_trailer(frame_info)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

BATCH_SIZE = 4
EPOCHS = 5
TRAIN_SAMPLES = len(trainX)
VAL_SAMPLES = len(valX)

# now we have to explicitly state shape of our samples because of generators gah
# (x, y, z, color)
train_shape = (longest_trailer_length,64,64,3) # maybe it will look like this idno the second to last is the idno part

model = Sequential()
model.add(Conv3D(32, (3, 3, 3), padding="same",input_shape=train_shape))
model.add(Activation("relu"))

model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # it is important to flatten your 2d tensors to 1d when going to FC-layers
model.add(Dense(512, bias_initializer='ones'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(len(genres)))
model.add(Activation("softmax"))

opt = rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

training_generator = data_generator(trainX, trainY, BATCH_SIZE, z_num=longest_trailer_length)
validation_generator = data_generator(valX, valY, BATCH_SIZE, z_num=longest_trailer_length)

model.fit_generator(
    generator = training_generator,
    steps_per_epoch = (TRAIN_SAMPLES // BATCH_SIZE),
    epochs = EPOCHS,
    verbose = 1,
    validation_data = validation_generator,
    validation_steps = (VAL_SAMPLES // BATCH_SIZE),
    use_multiprocessing = True,
    workers = 2,
    max_queue_size = 5
)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(predictions.shape)
print(testY.shape)
for i, row in enumerate(predictions):
    print(predictions[i])
    print(testY[i])
    break
# print(classification_report(testY.argmax(axis=1),
# 	       predictions.argmax(axis=0), target_names=genres))

# Store the model on disk.
#print("[INFO] serializing and storing the model ...")
#model.save(args["model"])
