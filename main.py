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
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = grayscale[y:y + h, x:x + w]
        face_resize = cv2.resize(face,(width,height))
        prediction = recogniser.predict(face_resize)
        print(prediction)
        if prediction[1]<500:
            cv2.putText(img,"%s - %.0f" %(names[prediction[0]], prediction[1]), (x-10 , y-10), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
        else:
            cv2.putText(img, "not recognised" ,(x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0))
             
         
    cv2.imshow("OpenCV", img) 

    key = cv2.waitKey(10)
    #space key
    if key == 27:
        break
