# Sign Language Interpreter 

A recent estimate claims that around 13 million people have some level of proficiency in American Sign Language, making it the third most commonly used language in the United States. 

The goal of our project is to create a system that will convert video feed of a person signing in American Sign Language into English text. We hope to encourage communication between those who have no other way of expressing their ideas other than using Sign Language and those who do not know how to sign. We hope to cater to the 500,000 deaf people in the US and Canada who’s natural language is the American Sign Language (ASL), and work towards creating a more inclusive environment.

We use a combination of Convolutional Neural Networks and LSTMs to tackle this problem. 

Find dependencies [here](https://docs.google.com/document/d/1NtFzyRzfd4Q186DdbciX0jrTyG6WlmhNEKeaMKJd7DU/edit)

In order to run this module, please clone the repository and perform the following
1. Download the Raw Data of the LSA64: A Dataset for Argentinian Sign Language [here](https://mega.nz/#!kJBDxLSL!zamibF1KPtgQFHn3RM0L1WBuhcBUvo0N0Uec9hczK_M): and place into newly created folder called "test" in the project's code directory.

2. Run ``python sort_files.py``. This will arrange all of the raw data into folders corresponding to each word category. Please change the appropriate paths in the file. 

3. Run python ``python video_to_frame.py``. This will capture 200 frames for each raw video as images and store them in the appropriate folder.

4. We will now apply Transfer Learning to add these new categories of Sign Language words to the pretrained Inception model. Download "retrain.py" [here](https://raw.githubusercontent.com/tensorflow/hub/r0.1/examples/image_retraining/retrain.py). Note: This link may change in the future. 

Run the following command:

``nohup python retrain.py --bottleneck_dir=bottlenecks --model_dir=inception --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=majorData &``











