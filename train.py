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

with open('frame_info.json', 'r') as f:
    frame_info = json.load(f)

# get the data
df = pd.read_csv('preprocessed_movies.csv')
movieids = [f[0] for f in frame_info['good']]
prune_mask = df['movieId'].isin(movieids)
df = df.loc[prune_mask]

genres = df.columns[4:].tolist()
print(df.columns)
print(df.columns[4:])
str_length = max(len(x) for x in genres)
sep_length = len(genres) * str_length
frame_sequences = []
labels = []

AMOUNT_TO_TRAIN = 100
LIMIT_TRAIN_SET = True

# drop all data not needed for machine learning
if LIMIT_TRAIN_SET:
    data = df.loc[:AMOUNT_TO_TRAIN,'movieId']
    labels = df.loc[:AMOUNT_TO_TRAIN, genres].values.tolist()
else:
    data = df.loc[:,'movieId']
    labels = df.loc[:, genres].values.tolist()

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

BATCH_SIZE = 4
EPOCHS = 2
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

training_generator = data_generator(trainX, trainY, BATCH_SIZE)
validation_generator = data_generator(valX, valY, BATCH_SIZE)
testing_generator = data_generator(testX, testY, BATCH_SIZE)


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
plt.show()

print("[INFO] evaluating network...")
#predictions = model.predict_generator(
#    testing_generator,
#    steps=(TEST_SAMPLES // BATCH_SIZE)+1,
#    max_queue_size=5,
#    workers=THREADS,
#    use_multiprocessing=MULTI_THREAD,
#    verbose=1
#)
#counter = 0
#for pred in predictions:
#    if counter >= len(predictions):
#        break
#    proba = pred
#    #idxs = np.argsort(proba)[::-1][:2]
#    print(" ".join([s.rjust(str_length) for s in genres]))
#    print(" ".join([("{:.2f}".format(p*100)).rjust(str_length) for p in proba]))
#    print(" ".join([str(v).rjust(str_length) for v in testY[counter]]))
#    print("="*sep_length)
#    counter += 1
