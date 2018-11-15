import cv2
import os
import pickle
from os.path import join, exists
from preprocessing import segment_hand as hs

from random import shuffle

from math import floor

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def split_test_train(datadir):
    """
    Moves raw video data into training (70%) and testing (30%) sets 
    """

    all_files = mylistdir(os.path.abspath(datadir))

    for file in all_files:
        # Get a list of the files
        file_direc = datadir + file
        if(os.path.isdir(file_direc)):
            main_dir = os.path.abspath(datadir + file)
            data_files = list(filter(lambda file: file.endswith('mp4'), mylistdir(main_dir)))

            # Randomize the files
            shuffle(data_files)

            #Split files into training and testing sets
            split = 0.7
            split_index = floor(len(data_files) * split)
            training = data_files[:split_index]
            testing = data_files[split_index:]

            train_dir = main_dir + "/train/"
            test_dir = main_dir + "/test/"

            if(not os.path.exists(train_dir)):
                os.makedirs(train_dir)
            if(not os.path.exists(test_dir)):
                os.makedirs(test_dir)

            for file in training:
                from_dir = main_dir + "/" + file
                to_dir =  train_dir + file
                shutil.move(from_dir, to_dir)

            for file in testing:
                from_dir = main_dir + "/" + file
                to_dir =  test_dir + file
                shutil.move(from_dir, to_dir)


def convert_to_frames(dataset,word_count,input_type,output_pickle_name):
    """
    Takes Raw training dataset and converts them from video to pictures taken at 200 frames 
    """
    pickle_file = []
    frame_count = 0

    # Create folder to store frames for all words 
    rootPath = os.getcwd()
    # need to change image data for different conversions 
    image_data = os.path.join(os.getcwd(), "preprocessing/image_data_test")
    if (not exists(image_data)):
        os.makedirs(image_data)

    # Create data file that will store pickle file 
    data = os.path.join(os.getcwd(),"pickle_data")
    if (not exists(data)):
        os.makedirs(data)

    dataset = os.path.join(os.getcwd(), dataset)
    os.chdir(dataset)

    # Get all files with raw data for words, only keep how many you want
    gesture_list = mylistdir(os.getcwd())
    gesture_list = gesture_list[:word_count]

    for gesture in gesture_list:
        gesture_path = os.path.join(dataset, gesture)
        os.chdir(gesture_path)

        # Create directory to store images
        frames = os.path.join(image_data, gesture)
        if(not os.path.exists(frames)):
            os.makedirs(frames)
	
        videos = mylistdir(os.getcwd() + input_type)
        videos = [video for video in videos if(os.path.isfile(os.getcwd() + input_type + video))]

        for video in videos:
            name = os.getcwd() + input_type + video
            frame_count = frame_count + 1
            print(str(frame_count) +  " : " +  str(name))

            # capturing input video
            cap = cv2.VideoCapture(name)  
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print("Frame Count",frameCount)
            count = 0
            os.chdir(frames)
            lastFrame = None
            while(1):
                # extract frame
                ret, frame = cap.read()  
                if ret is False:
                    break

                framename = os.path.splitext(video)[0]
                framename = framename+"_frame_"+str(count)+".jpeg"
                pickle_file.append([join(frames, framename), gesture, frameCount])

                if(not os.path.exists(framename)):
                    frame = hs.hand_segment(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lastFrame = frame
                    cv2.imwrite(framename, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                count += 1

            while(count < 201):
                framename = os.path.splitext(video)[0]
                framename = framename+"_frame_"+str(count)+".jpeg"
                pickle_file.append([join(frames, framename), gesture, frameCount])
                if(not os.path.exists(framename)):
                    cv2.imwrite(framename, lastFrame)
                count += 1

            os.chdir(gesture_path)
            cap.release()
            cv2.destroyAllWindows()

    print(pickle_file)

    os.chdir(rootPath)
    with open(output_pickle_name, 'wb') as handle:
        pickle.dump(pickle_file, handle, protocol=2)

if __name__ == '__main__':
    #split_test_train("raw_data/")
    convert_to_frames("preprocessing/raw_data/",10,"/train/")

  
