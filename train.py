from collections import defaultdict
import numpy as np
import pandas as pd
from keras.optimizers import SGD, rmsprop, Adam, Nadam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import cv2
import csv

imgs = []
processed_movies = set()

labels = []
# Read frames

movie_to_framecount = {}

for i, movie in enumerate(os.listdir("frames")):
    print("processing movie {}/{}".format(i, len(os.listdir("frames"))), end="\r")
    processed_movies.add(movie)
    movie_to_framecount[movie] = len(os.listdir("frames/{}".format(movie)))
    for frame in os.listdir("frames/%s" % movie):
        frame_data = cv2.imread("frames/%s/%s" % (movie, frame))
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        frame_data = cv2.resize(frame_data, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        imgs.append(frame_data)

genre_lookup = defaultdict(list)


genres = None
labels = []
# Read genres, i.e. labels
with open("preprocessed_movies.csv") as movie_file:
    reader = csv.DictReader(movie_file)

    for row in reader:
        if row['movieId'] not in processed_movies:
            continue
        if not genres:
            genres = reader.fieldnames[4:]
        thisgenre = []
        for genre in reader.fieldnames[4:]:
            thisgenre.append(int(row[genre]))
        for _ in range(movie_to_framecount[row['movieId']]):
            labels.append(thisgenre)

#print(genre_lookup['1']['Adventure'])
# labels = genre_lookup.values()
#print(genre_lookup.keys())
#print(labels)
imgs = np.array(imgs)
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(imgs, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",input_shape=trainX.shape[1:]))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # it is important to flatten your 2d tensors to 1d when going to FC-layers
model.add(Dense(512, bias_initializer='ones'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(len(genres)))
model.add(Activation("softmax"))


EPOCHS = 5

print(len(labels[0]))
opt = rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Now train the ANN ...
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)
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