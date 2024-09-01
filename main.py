import cv2
import numpy 
import os

dataset = "C:/Users/Adhay/OneDrive/Desktop/Open CV/Lesson 9/dataset"
harfile = "C:/Users/Adhay/OneDrive/Desktop/Open CV/Lesson 9/haarcascade_frontalface_default.xml"

#create a list of images and a list of corresponding names
(images, labels,names,id) = ([],[],{},0)

for(subdirs, dirs, files) in os.walk (dataset):
    for subdir in dirs:
        names[id] = subdir
        print(names)
        subject_path = os.path.join(dataset,subdir)
        print(subject_path)
        for filename in os.listdir(subject_path):
            path = subject_path + "/" + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id = id + 1    


#Create a Numpy array from the two lists above 
(images,labels) = [numpy.array(lis) for lis in [images,labels]]

(width,height) = 130,110

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.train(images,labels)

face_cascade = cv2.CascadeClassifier(harfile)

webcam = cv2.VideoCapture(0)

while True:
    ret,img = webcam.read()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(grayscale,1.3,5) 
