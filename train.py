#!/usr/bin/python3
from collections import defaultdict
import numpy as np
import pandas as pd
from keras.optimizers import SGD, rmsprop, Adam, Nadam
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import cv2
import csv
import json
import sys
import keras.backend as K
import tensorflow as tf
import datetime
import argparse
import time
from subprocess import call

# size of sampled image
sz = 64

# its win32, maybe there is win64 too?
is_windows = sys.platform.startswith('win')

MULTI_THREAD = True
THREADS = 2
if is_windows:
    MULTI_THREAD = False
    THREADS = 1


# FOR NOW THIS CONSTANT IS NON DYNAMIC BASED ON LOADED MODEL
# PROBABLY SHOULD DO STUFF TO MAKE IT DYNAMIC IN THE PROCESS
# YT LINK FUNCTION!
SLICE_SIZE = 10

def threshold_accuracy(y_true, y_pred):
    """
    Each prediction is considered invalid if the confidence lies between the interval (0.5-threshold, 0.5+threshold).
    Returns (valid correct predictions / valid predictions)
    """
    threshold = 0.25
    lower_threshold = 0.5-threshold
    upper_threshold = 0.5+threshold

    valid_filter = lambda v: v[np.where(np.logical_or(v < lower_threshold, v > upper_threshold))]


    y_pred_valid = tf.py_func(valid_filter, [y_pred], tf.float32)
    y_pred_rounded = tf.to_float(K.round(y_pred))
    y_true = tf.to_float(y_true)
    y_diff = K.equal(y_true, y_pred_rounded)
    y_diff = tf.cast(y_diff, tf.float32)

    return K.sum(y_diff)/K.sum(K.ones_like(y_pred_valid))

def threshold_accuracy_lists(y_true, y_pred):
    threshold = 0.25
    lower_threshold = 0.5-threshold
    upper_threshold = 0.5+threshold

    valid_preds = [i for i,v in enumerate(y_pred) if v < lower_threshold or v > upper_threshold]
    corrects = [i for i in valid_preds if round(valid_preds[i]) == y_true[i]]
    return len(corrects)/len(valid_preds)

def threshold_accuracy_2d_lists(y_true, y_pred, lower_threshold, upper_threshold):
    correct = 0
    total = 0
    incorrect = 0
    total_ones = 0
    correct_ones = 0
    non_guesses = 0
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            p = y_pred[i,j]
            if p > upper_threshold:
                total_ones += 1
                if round(p) == y_true[i][j]:
                    correct_ones += 1
            if p < lower_threshold or p > upper_threshold:
                if round(p) == y_true[i][j]:
                    correct += 1
                else:
                    incorrect += 1
                total += 1
            else:
                non_guesses += 1
    if total == 0:
        return "The model made no guesses"
    return "Percent correct {} | Correct {} | Incorrect {} | Precent Guessed {} | guesses {} | non-guesses {} | correct-ones {} | total-ones {} ".format(correct/total,
                                                                                                                       correct,
                                                                                                                       incorrect,
                                                                                                                       total/(len(y_true) * len(y_true[0])),
                                                                                                                       total,
                                                                                                                       non_guesses,
                                                                                                                       correct_ones,
                                                                                                                       total_ones)


#t_y_true = tf.constant([0, 1, 0])
#t_y_pred = tf.constant([0.23, 0.1, 0.1])

#print(threshold_accuracy(t_y_true, t_y_pred))
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
                frame_data = cv2.resize(frame_data, dsize=(sz, sz), interpolation=cv2.INTER_NEAREST)
                #frame_data = cv2.blur(frame_data, (5, 5))
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
                frame_data = cv2.resize(frame_data, dsize=(sz, sz), interpolation=cv2.INTER_NEAREST)
                frames.append(frame_data)
                if len(frames) >= self.slice_sizes:
                    break
            for _ in range(self.slice_sizes-len(frames)):
                frames.append(np.zeros((sz,sz,3)).astype(np.uint8))
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
def create_model():
    
    genres = None
    with open("genres.json", "r") as f:
        genres = json.load(f)

    train_shape = (SLICE_SIZE,sz,sz,3) # maybe it will look like this idno the second to last is the idno part
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), padding="same",input_shape=train_shape))
    model.add(Activation("relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv3D(32, (3, 3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv3D(32, (3, 3, 3), padding="same"))
    model.add(Activation("relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Conv3D(16, (3, 3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv3D(16, (3, 3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv3D(16, (3, 3, 3), padding="same"))
    model.add(Activation("relu"))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    # model.add(Conv3D(8, (3, 3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(Conv3D(8, (3, 3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(Conv3D(8, (3, 3, 3), padding="same"))
    # model.add(Activation("relu"))

    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten()) # it is important to flatten your 2d tensors to 1d when going to FC-layers
    model.add(Dense(1024, bias_initializer='ones'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(1024, bias_initializer='ones'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dense(1024, bias_initializer='ones'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(len(genres)))
    model.add(Activation("sigmoid"))

    opt = rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def train_model(model):
    with open('frame_info.json', 'r') as f:
        frame_info = json.load(f)

    # get the data
    df = pd.read_csv('preprocessed_better_dist.csv')
    movieids = [f[0] for f in frame_info['good']]
    prune_mask = df['movieId'].isin(movieids)
    df = df.loc[prune_mask]
    df['movieId'] = pd.to_numeric(df['movieId'])

    genres = df.columns[4:].tolist()

    AMOUNT_TO_TRAIN = 500
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
    movieId_sliceN_dict = {}
    for mid, fcount in frame_info['good']:
        movieId_sliceN_dict[int(mid)] = int(np.ceil(fcount/np.float(SLICE_SIZE)))

    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    BATCH_SIZE = 64
    EPOCHS = 20
    TRAIN_SAMPLES = len(trainX)
    VAL_SAMPLES = len(valX)
    TEST_SAMPLES = len(testX)

    training_generator = slice_generator(trainX, trainY, BATCH_SIZE, movieId_sliceN_dict, movieId_label_dict,SLICE_SIZE)
    validation_generator = slice_generator(valX, valY, BATCH_SIZE, movieId_sliceN_dict, movieId_label_dict,SLICE_SIZE)
    testing_generator = slice_generator(testX, testY, BATCH_SIZE, movieId_sliceN_dict, movieId_label_dict,SLICE_SIZE)

    if type(model) == str:
        model = load_model(model_path)
    elif type(model) != Sequential:
        print("this shit is no bueno")
        exit()

    print(model.summary())

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
    predictions = model.predict_generator(
        testing_generator,
        steps=(TEST_SAMPLES // BATCH_SIZE)+1,
        max_queue_size=5,
        workers=THREADS,
        use_multiprocessing=MULTI_THREAD,
        verbose=1
    )

    metrics = threshold_accuracy_2d_lists(testY, predictions, 0.1, 0.5)
    print(metrics)
    model_string = "model-slicesize_{}-{}.h5".format(SLICE_SIZE ,datetime.datetime.now().strftime("%m-%d-%H%M%S"))
    model.save(model_string)

def frame_folder_to_slices(folder):
    # Log the time
    time_start = time.time()
    count = 0
    actual_frames = 0
    slices = []
    curr_slice = []
    video_length = len(os.listdir(folder))-1
    print("vide_length: %s" % video_length)
    for frame in sorted(os.listdir(folder)):
        print("{}/{}".format(folder, frame))
        # Extract the frame
        frame_img = cv2.imread("{}/{}".format(folder, frame))
        frame_img = cv2.resize(frame_img,dsize=(sz, sz), interpolation=cv2.INTER_NEAREST)
        actual_frames += 1
        curr_slice.append(frame_img)
        if len(curr_slice) >= SLICE_SIZE:
            slices.append(curr_slice)
            curr_slice = []
        # If there are no more frames left
        if (actual_frames > video_length):
            if len(curr_slice) > 0:
                for _ in range(SLICE_SIZE-len(curr_slice)):
                    curr_slice.append(np.zeros((sz,sz,3)).astype(np.uint8))
                slices.append(curr_slice)
            # Log the time again
            time_end = time.time()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break
    print(slices) 
    return slices

def video_to_slices(video_path):
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(video_path)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("video length was {} should end in {} slices".format(video_length, int(np.ceil(video_length/np.float(SLICE_SIZE)))))
    count = 0
    actual_frames = 0
    print ("Converting video..\n")
    slices = []
    curr_slice = []
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        frame = cv2.resize(frame,dsize=(sz, sz), interpolation=cv2.INTER_NEAREST)
        actual_frames += 1
        curr_slice.append(frame)
        if len(curr_slice) >= SLICE_SIZE:
            slices.append(curr_slice)
            curr_slice = []
        # If there are no more frames left
        if (actual_frames > video_length):
            if len(curr_slice) > 0:
                for _ in range(SLICE_SIZE-len(curr_slice)):
                    curr_slice.append(np.zeros((sz,sz,3)).astype(np.uint8))
                slices.append(curr_slice)
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break
    return slices

def process_youtube_link(model_path, youtube_link, confidence=0.6):

    genres = None
    with open("genres.json", "r") as f:
        genres = json.load(f)

    model = load_model(model_path)
    data_folder = './data'
    yt_vid_name = youtube_link.split('=')[-1]
    vid_output = "{}/{}".format(data_folder,yt_vid_name)
    call(["youtube-dl", "-o{}".format(vid_output), youtube_link, "--restrict-filenames", "-f", "mp4/worstvideo"])
    slices = video_to_slices(vid_output)
    print("slice info")
    print("len ",len(slices))
    print('type', type(slices))
    #print("shape ",np.array(slices).shape)
    #call(["rm", "-rf", vid_output])
    print(model.summary())
    genre_occ_dict = {key:0 for key in genres}
    for s in slices:
        p = model.predict(np.array([s]))
    
        for i,v in enumerate(p[0]):
            if v > confidence:
                genre_occ_dict[genres[i]] += 1
    print(genre_occ_dict)

def process_frame_folder(model_path, folder_path, confidence=0.6):

    genres = get_genres()

    model = load_model(model_path)
    #data_folder = './data'
    #vid_output = "{}/{}".format(data_folder,folder)
    slices = frame_folder_to_slices(folder_path)
    print("slice info")
    print("len ",len(slices))
    print('type', type(slices))
    #print("shape ",np.array(slices).shape)
    #call(["rm", "-rf", vid_output])
    print(model.summary())
    genre_occ_dict = {key:0 for key in genres}
    combined = []
    for s in slices:
        p = model.predict(np.array([s]))
        combined.append(p)

    np_combined = np.array(combined)
    mean_frame_confidences = np.average(np_combined, axis=0)
    print(convert_confidence_matrix_to_string(mean_frame_confidences)) 
    print()
    #    print(p)
    #    for i,v in enumerate(p[0]):
    #        if v > confidence:
    #            genre_occ_dict[genres[i]] += 1
    #print(genre_occ_dict)

def get_genres():
    genres = None
    with open("genres.json", "r") as f:
        genres = json.load(f)
    return genres

def convert_confidence_matrix_to_string(mat):
    genres = get_genres()
    genre_dict = {genre:0 for genre in genres}
    for i, genre in enumerate(genres):
        genre_dict[genre] = mat[0][i]
    kv_tuples = sorted([(k, v) for k, v in genre_dict.items()], key = lambda x: x[1], reverse = True)
    kv_strings = ["{}: {:.3f}".format(k, v) for k,v in kv_tuples]
    return "\n".join(kv_strings)
    
def continue_training(model_path):
    model = load_model(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A deep convolutional neural network for inference on film trailers")
    parser.add_argument('-m', '--modelname', metavar='model_name', type=str, help='Path to the model you want to use')
    parser.add_argument('-yt','--youtubelink', metavar='youtube_link', type=str, help='A link to a youtubevideo to be processed')
    parser.add_argument('-f','--folder', metavar='folder', type=str, help='A path to a folder of frames, useful for short sequences not hosted on youtube')
    args = parser.parse_args()
    model_path = args.modelname
    youtube_link = args.youtubelink
    folder = args.folder
    print(args)
    if not model_path:
        model = create_model()
        train_model(model)
    else:
        if youtube_link:
            process_youtube_link(model_path, youtube_link)
        elif folder:
            process_frame_folder(model_path, folder)
        # if no option was selected, continue to train supplied model
        else:
            train_model(model_path)


    #if not model_path and not youtube_link:
    #    model = create_model()
    #    train_model(model)
    #elif not youtube_link:
    #    train_model(model_path)
    #elif youtube_link:
    #    process_youtube_link(model_path, youtube_link)
    #elif folder:
    #    process_frame_folder(model_path)





