import shutil
import random
import numpy as np
import cv2
import argparse
import sys
import time

from scripts.label_image import *
from pipeline.video_to_frame import *
from pipeline.mode_accuracy import *

import os
from flask import Flask, render_template, Response,request
import json


app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/run_demo', methods=['GET'])
def determine_sign():
    """Translate sign into word given signal to do so"""

    file_name = "RecordedVideo.webm"
    direct_name = "segmented_frames/"
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    # Uncomment 'model_file' and 'label_file' below to switch to 64 word model:
    #model_file = "tf_files/retrained_graph2.pb"
    #label_file = "tf_files/retrained_labels2.txt"

    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    graph = load_graph(model_file)
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    sample_size = 20 # number of images to randomly sample from the processed video

    if not os.path.exists(direct_name):
        os.makedirs(direct_name)

    downloads_dir = "/Users/Marwa/Downloads/"

    predicted_word = run_pipeline(downloads_dir + file_name,direct_name ,model_file, label_file, input_height, input_width, input_mean, input_std, input_layer, output_layer, graph, input_name, output_name, input_operation, output_operation, sample_size)

    if os.path.exists(downloads_dir + file_name):
        os.remove(downloads_dir + file_name)
    #remove segmented frames from directory in case you're 
    #running multiple videos sequentially 

    # shutil.rmtree(direct_name)
    # os.mkdir(direct_name)

    return json.dumps({"word":predicted_word})

@app.route('/sample_video', methods=['GET'])
def sample_determine_sign():
    """Translate sign into word given signal to do so"""

    file_name = "accept4.mov"
    direct_name = "segmented_frames/"
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    # Uncomment 'model_file' and 'label_file' below to switch to 64 word model:
    #model_file = "tf_files/retrained_graph2.pb"
    #label_file = "tf_files/retrained_labels2.txt"

    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    graph = load_graph(model_file)
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    sample_size = 20 # number of images to randomly sample from the processed video

    if not os.path.exists(direct_name):
        os.makedirs(direct_name)

    inner_dir = "/Users/Marwa/Documents/TestVideos/"

    predicted_word = run_pipeline(inner_dir + file_name,direct_name ,model_file, label_file, input_height, input_width, input_mean, input_std, input_layer, output_layer, graph, input_name, output_name, input_operation, output_operation, sample_size)

    #remove segmented frames from directory in case you're 
    #running multiple videos sequentially 

    # shutil.rmtree(direct_name)
    # os.mkdir(direct_name)

    return json.dumps({"word":predicted_word})

def run_pipeline(video_path, frame_dir, model_file, label_file, input_height, input_width, 
                    input_mean, input_std, input_layer, output_layer, graph, input_name, output_name, 
                    input_operation, output_operation, sample_size):

    """Segments the video, passes through CNN, 
    and gets predicted word to be spoken"""

    segment_video(video_path, frame_dir)

    predicted_word = predict_word(frame_dir, model_file, label_file, input_height, input_width, input_mean, input_std, input_layer, output_layer, graph, input_name, output_name, input_operation, output_operation, sample_size)

    return predicted_word











