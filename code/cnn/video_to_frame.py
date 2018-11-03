import cv2
import os
import pickle
from os.path import join, exists
import segment_hand as hs

def convert_to_frames(dataset):
    """
    Takes Raw dataset and converts them from video to pictures taken at 200 frames 
    """
    pickle_file = []
    frame_count = 0

    # Create folder to store frames for all words 
    rootPath = os.getcwd()
    majorData = os.path.join(os.getcwd(), "majorData")
    if (not exists(majorData)):
        os.makedirs(majorData)

    # Create data file that will store pickle file 
    data = os.path.join(os.getcwd(),"data")
    if (not exists(majorData)):
        os.makedirs(data)

    dataset = os.path.join(os.getcwd(), dataset)
    os.chdir(dataset)
    # Get all files with raw data for words 
    x = os.listdir(os.getcwd())

    for i in range(len(x)): 
        if x[i] == "test":
            x = x[i]

    # Looking through every word's file 
    for gesture in x:
        word = gesture
        gesture = os.path.join(dataset, gesture)
        os.chdir(gesture)
        frames = os.path.join(majorData, word)
        if(not os.path.exists(frames)):
            os.makedirs(frames)
        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(video))]

        for video in videos:
            name = os.path.abspath(video)
            frame_count = frame_count+1
            print(frame_count, " : ", name)
            cap = cv2.VideoCapture(name)  # capturing input video
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(frameCount)
            count = 0
            os.chdir(frames)
            lastFrame = None
            while(1):
                ret, frame = cap.read()  # extract frame
                if ret is False:
                    break
                framename = os.path.splitext(video)[0]
                framename = framename+"_frame_"+str(count)+".jpeg"
                pickle_file.append([join(frames, framename), word, frameCount])

                if(not os.path.exists(framename)):
                    frame = hs.handsegment(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lastFrame = frame
                    cv2.imwrite(framename, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                count += 1
            while(count < 201):
                framename = os.path.splitext(video)[0]
                framename = framename+"_frame_"+str(count)+".jpeg"
                pickle_file.append([join(frames, framename), word, frameCount])
                if(not os.path.exists(framename)):
                    cv2.imwrite(framename, lastFrame)
                count += 1

            os.chdir(gesture)
            cap.release()
            cv2.destroyAllWindows()
    print(pickle_file)

    os.chdir(rootPath)
    with open('data/labeled-frames-1.pkl', 'wb') as handle:
    # with open('data/labeled-frames-2.pkl', 'wb') as handle:
        pickle.dump(pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


convert_to_frames("test/")


