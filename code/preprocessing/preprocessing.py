import sys 

import cv2
import os
import pickle
from os.path import join, exists
import segment_hand as hs

from random import shuffle

from math import floor
from os import listdir
from os.path import isfile, join

import os
import shutil
def sort_files(data_dir):

    home_directory = os.getcwd() + data_dir
    files = [f for f in listdir(home_directory) if isfile(join(home_directory, f))]

    name_dict = {"Opaque": "001", 
                 "Light-Red": "002", 
                 "Green": "003", 
                 "Yellow": "004",
                 "Bright": "005",
                 "Light-blue": "006",
                 "Colors": "007",
                 "Red": "008",
                 "Women": "009", 
                 "Enemy": "010",
                 "Son": "011",
                 "Man": "012",
                 "Away": "013",
                 "Drawer": "014",
                 "Born": "015",
                 "Learn": "016",
                 "Call": "017",
                 "Skimmer": "018",
                 "Bitter": "019",
                 "Sweet milk": "020",
                 "Milk": "021",
                 "Water": "022",
                 "Food": "023",
                 "Argentina": "024",
                 "Uruguay": "025",
                 "Country": "026",
                 "Last name": "027",
                 "Where":"028",
                 "Mock": "029",
                 "Birthday": "030",
                 "Breakfast": "031",
                 "Photo": "032",
                 "Hungry": "033",
                 "Map": "034",
                 "Coin": "035",
                 "Music": "036",
                 "Ship": "037",
                 "None": "038",
                 "Name": "039",
                 "Patience": "040",
                 "Perfume": "041",
                 "Deaf": "042",
                 "Trap": "043",
                 "Rice": "044",
                 "Barbecue": "045",
                 "Candy": "046",
                 "Chewing-gum": "047",
                 "Spaghetti": "048",
                 "Yogurt": "049",
                 "Accept": "050",
                 "Thanks": "051",
                 "Shut down": "052",
                 "Appear": "053",
                 "To land": "054",
                 "Catch": "055",
                 "Help": "056",
                 "Dance": "057",
                 "Bathe": "058",
                 "Buy": "059",
                 "Copy": "060",
                 "Run": "061",
                 "Realize": "062",
                 "Give": "063",
                 "Find": "064"}

    for name, number in name_dict.items():
        output_names = [f for f in files if (f[0:3] == number)]
        for file_name in output_names: 
            if not os.path.exists(home_directory + "/" +  name):
                os.mkdir(home_directory + "/" + name)
            current_directory = home_directory + "/" + file_name
            print(current_directory)
            new_directory = home_directory + "/" +  name + "/" + file_name

            shutil.move(current_directory, new_directory)
            print(name, " Moved!")

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def split_test_train(main_dir):
    """
    Moves raw video data into training (70%) and testing (30%) sets 
    """
    main_dir = os.getcwd() + main_dir
    data_dir = main_dir + "all/"

    all_files = mylistdir(os.path.abspath(data_dir))
    for file_name in all_files:
        # Get a list of the files
        file_direc = os.path.abspath(data_dir + file_name)
        if(os.path.isdir(file_direc)):
            data_files = list(filter(lambda file: file.endswith('mp4'), mylistdir(file_direc)))

            # Randomize the files
            shuffle(data_files)

            #Split files into training and testing sets
            split = 0.7
            split_index = floor(len(data_files) * split)
            training = data_files[:split_index]
            testing = data_files[split_index:]

            train_dir = main_dir + "train/"
            test_dir = main_dir + "test/"

            if(not os.path.exists(train_dir)):
                os.makedirs(train_dir)

            if(not os.path.exists(test_dir)):
                os.makedirs(test_dir)
	    
            for file in training:
                from_dir = data_dir + file_name + "/" + file
                to_dir =  train_dir + file_name 
                if(not os.path.exists(to_dir)):
                    os.makedirs(to_dir)

                to_dir += "/" + file
                shutil.move(from_dir, to_dir)

            for file in testing:
                from_dir = data_dir + file_name + "/" + file
                to_dir =  test_dir + file_name 
                if(not os.path.exists(to_dir)):
                    os.makedirs(to_dir)

                to_dir += "/" + file
                shutil.move(from_dir, to_dir)

            os.rmdir(file_direc)
            print("Done Splitting Dataset")

def convert_to_frames(dataset,word_count,input_type,output_pickle_name):
    """
    Takes Raw training dataset and converts them from video to pictures taken at 200 frames 
    """
    pickle_file = []
    frame_count = 0

    # Create folder to store frames for all words 
    rootPath = os.getcwd()
    # need to change image data for different conversions 
    image_data = os.path.join(os.getcwd(), "preprocessing/image_data_" + input_type)
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
    #gesture_list = gesture_list[:word_count]

    for gesture in gesture_list:
        gesture_path = os.path.join(dataset, gesture)
        os.chdir(gesture_path)

        # Create directory to store images
        frames = os.path.join(image_data, gesture)
        if(not os.path.exists(frames)):
            os.makedirs(frames)
	
        videos = mylistdir(os.getcwd())
        print(os.getcwd())

        videos = [video for video in videos if(os.path.isfile(os.getcwd() + '/' +  video))]
        for video in videos:
            name = os.getcwd() + '/' +  video
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
    sort_dir = "/preprocessing/all/"
    sort_files(sort_dir)

    split_dir = "/preprocessing/"
    split_test_train(split_dir)
    convert_to_frames("preprocessing/train/",10,"train","pickle_train.pkl")

  
