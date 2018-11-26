# Sign Language Interpreter 

A recent estimate claims that around 13 million people have some level of proficiency in American Sign Language, making it the third most commonly used language in the United States. 

The goal of our project is to create a system that will convert video feed of a person signing in American Sign Language into English text. We hope to encourage communication between those who have no other way of expressing their ideas other than using Sign Language and those who do not know how to sign. We hope to cater to the 500,000 deaf people in the US and Canada who’s natural language is the American Sign Language (ASL), and work towards creating a more inclusive environment.

We use a combination of Convolutional Neural Networks and LSTMs to tackle this problem. 

## Getting Started
To install the dependencies for this project, run the following commands: 

``pip install opencv-python``

``pip install "tensorflow>=1.7.0``

``pip install tensorflow-hub``

``pip install tflearn``

``pip install pickle``

``pip install numpy``

## Pipeline

In order to run this module, please clone the repository and perform the following
1. Download the Raw Data of the LSA64: A Dataset for Argentinian Sign Language [here](https://mega.nz/#!kJBDxLSL!zamibF1KPtgQFHn3RM0L1WBuhcBUvo0N0Uec9hczK_M). Place the unzipped file 'all' inside of a new directory 'preprocessing'.

3. Run ``python preprocessing.py``. This will arrange all of the raw data into folders corresponding to each word category, split your dataset into a training and a test set (70:30 ratio), and capture 200 frames for each raw video in the training set as images and store them in the appropriate folder. Note that can you can specify how many words to include in your training set.

4. We will now apply Transfer Learning to add these new categories of Sign Language words to the pretrained Inception model. Download ``retrain.py`` [here](https://raw.githubusercontent.com/tensorflow/hub/r0.1/examples/image_retraining/retrain.py) and store in the "code" directory. Note: This link may change in the future. 

Run the following command:

``nohup python cnn/retrain.py --bottleneck_dir=bottlenecks --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=preprocessing/image_data --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1 &``

5. We will now run a command that will create a pickle file to feed into the LSTM model that includes the accuracy values for reach word in a matrix. 

`` nohup python cnn/generate_cnn_output.py & ``

