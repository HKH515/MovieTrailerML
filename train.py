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
import sys
# its win32, maybe there is win64 too?
is_windows = sys.platform.startswith('win')

MULTI_THREAD = True
THREADS = 2
if is_windows:
    MULTI_THREAD = False
    THREADS = 1


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
            for frame in sorted(os.listdir("frames/%s" % movieId)):
                frame_data = cv2.imread("frames/%s/%s" % (movieId, frame))
                frame_data = cv2.resize(frame_data, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
                frames.append(frame_data)
                if len(frames) >= self.z_num:
                    break
            batch_data.append(frames)
        return np.array(batch_data), np.array(batch_y)

class slice_generator(Sequence):
            
    def __init__(self, ids, labels, batch_size, movie_sliceN_dict,movieid_labels, slice_sizes):
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.movie_sliceN_dict = movie_sliceN_dict
        self.movieid_labels = movieid_labels
        self.slice_sizes = slice_sizes
        self.slice_map = self.create_slice_map()

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        batch_info = self.slice_map[idx]
        batch_data = []
        batch_labels = []
        for movieId,slices in batch_info:
            frames = []
            for frame in sorted(os.listdir("frames/%s" % movieId))[slices*self.slice_sizes:]:
                frame_data = cv2.imread("frames/%s/%s" % (movieId, frame))
                frame_data = cv2.resize(frame_data, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
                frames.append(frame_data)
                if len(frames) >= self.slice_sizes:
                    break
            for _ in range(self.slice_sizes-len(frames)):
                frames.append(np.zeros((64,64,3)).astype(np.uint8))
            batch_data.append(frames)
            batch_labels.append(self.movieid_labels[movieId])

        return np.array(batch_data), np.array(batch_labels)

    def create_slice_map(self):
        slice_map = []
        curr_batch = []
        for movieId in self.ids:
            for v in range(self.movie_sliceN_dict[movieId]):
                curr_batch.append((movieId, v))
                if len(curr_batch) >= self.batch_size:
                    slice_map.append(curr_batch)
                    curr_batch = []
        if len(curr_batch) != 0:
            slice_map.append(curr_batch)
        return slice_map


with open('frame_info.json', 'r') as f:
    frame_info = json.load(f)

# get the data
df = pd.read_csv('preprocessed_movies.csv')
movieids = [f[0] for f in frame_info['good']]
prune_mask = df['movieId'].isin(movieids)
df = df.loc[prune_mask]
df['movieId'] = pd.to_numeric(df['movieId'])

genres = df.columns[4:].tolist()
print(df.columns)
print(df.columns[4:])
str_length = max(len(x) for x in genres)
sep_length = len(genres) * str_length + len(genres)-1
frame_sequences = []
labels = []

AMOUNT_TO_TRAIN = 40
LIMIT_TRAIN_SET = False

# drop all data not needed for machine learning
if LIMIT_TRAIN_SET:
    data = df.loc[:AMOUNT_TO_TRAIN,'movieId']
    labels = df.loc[:AMOUNT_TO_TRAIN, genres].values.tolist()
else:
    data = df.loc[:,'movieId']
    labels = df.loc[:, genres].values.tolist()

# movieid lable dict (needed for the slice generator)
movieId_label_dict = dict(zip(data,labels))

# movieid to amount of slices
SLICE_SIZE = 10
movieId_sliceN_dict = {}
for mid, fcount in frame_info['good']:
    movieId_sliceN_dict[int(mid)] = int(np.ceil(fcount/np.float(SLICE_SIZE)))

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

BATCH_SIZE = 32
EPOCHS = 5
TRAIN_SAMPLES = len(trainX)
VAL_SAMPLES = len(valX)
TEST_SAMPLES = len(testX)

# now we have to explicitly state shape of our samples because of generators gah
# (x, y, z, color)
train_shape = (10,64,64,3) # maybe it will look like this idno the second to last is the idno part

model = Sequential()
model.add(Conv3D(32, (3, 3, 3), padding="same",input_shape=train_shape))
model.add(Activation("relu"))

model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv3D(32, (3, 3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv3D(32, (3, 3, 3), padding="same"))
model.add(Activation("relu"))

model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv3D(32, (3, 3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv3D(32, (3, 3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv3D(32, (3, 3, 3), padding="same"))
model.add(Activation("relu"))

model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Flatten()) # it is important to flatten your 2d tensors to 1d when going to FC-layers
model.add(Dense(1024, bias_initializer='ones'))
model.add(Activation("relu"))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(1024, bias_initializer='ones'))
model.add(Activation("relu"))
model.add(BatchNormalization())

model.add(Dense(len(genres)))
model.add(Activation("sigmoid"))

opt = rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print(model.summary())

#training_generator = data_generator(trainX, trainY, BATCH_SIZE)
#validation_generator = data_generator(valX, valY, BATCH_SIZE)
#testing_generator = data_generator(testX, testY, BATCH_SIZE)
training_generator = slice_generator(trainX, trainY, BATCH_SIZE, movieId_sliceN_dict, movieId_label_dict,SLICE_SIZE)
validation_generator = slice_generator(valX, valY, BATCH_SIZE, movieId_sliceN_dict, movieId_label_dict,SLICE_SIZE)
testing_generator = slice_generator(testX, testY, BATCH_SIZE, movieId_sliceN_dict, movieId_label_dict,SLICE_SIZE)

H = model.fit_generator(
    generator = training_generator,
    steps_per_epoch = (TRAIN_SAMPLES // BATCH_SIZE),
    epochs = EPOCHS,
    verbose = 1,
    validation_data = validation_generator,
    validation_steps = (VAL_SAMPLES // BATCH_SIZE),
    use_multiprocessing = MULTI_THREAD,
    workers = THREADS,
    max_queue_size = 100
)

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show(block=False)

print("[INFO] evaluating network...")
predictions = model.predict_generator(
    testing_generator,
    steps=(TEST_SAMPLES // BATCH_SIZE)+1,
    max_queue_size=5,
    workers=THREADS,
    use_multiprocessing=MULTI_THREAD,
    verbose=1
)
for i,pred in enumerate(predictions):
    if i >= len(testY):
        break

    proba = pred
    print(" ".join([s.rjust(str_length) for s in genres]))
    print(" ".join([("{:.2f}".format(p*100)).rjust(str_length) for p in proba]))
    print(" ".join([str(v).rjust(str_length) for v in testY[i]]))
    print("="*sep_length)
