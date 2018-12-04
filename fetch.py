from __future__ import unicode_literals
import os
from subprocess import call
import csv
import cv2
import time
import av

import argparse
import ffmpeg
import sys

def video_to_frames(input_loc, output_loc, frame_skip):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        Non
    """
    try:
        os.mkdir(output_loc)
    except OSError as e:
        print(e)
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    actual_frames = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = (None, None)
        for i in range(frame_skip):
            ret, frame = cap.read()
            actual_frames += 1
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count), frame)
        count += 1
        # Write the results back to output location.
        # If there are no more frames left
        if (actual_frames > (video_length) - frame_skip):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__ == "__main__":
    data_folder = sys.argv[1].strip()
    frame_skip = int(sys.argv[2])
    try:
        os.mkdir("frames")
    except:
        pass

    csv_file = open("preprocessed_movies.csv", 'r')

    csv_reader = csv.DictReader(csv_file, delimiter = ',', quotechar = '"')

    cntr = 0

    downloaded_links = set()

    for row in csv_reader:
        yt_link = "https://youtube.com/watch?v={}".format(row["youtubeId"])
        if yt_link in downloaded_links:
            continue
        vid_output = "{}/{}".format(data_folder, row["movieId"])
        call(["youtube-dl", "-o{}".format(vid_output), yt_link, "--restrict-filenames", "-f", "mp4/worstvideo"])

        downloaded_links.add(yt_link)

        output_path = "%s/frames/%s" % (os.getcwd(), row["movieId"])
        video_to_frames(vid_output, output_path, frame_skip)
        call(["rm", "-rf", vid_output])

        if cntr > 200:
            exit()