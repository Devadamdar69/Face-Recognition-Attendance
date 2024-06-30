import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
myList= os.listdir(path)

images=[cv2.imread(f"{path}\{cl}") for cl in myList]
names=[i.split('.')[0] for i in myList]

def encoding(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodelist.append(face_recognition.face_encodings(img)[0])

    return encodelist

with open('Attendance.csv','r+') as f:
         f.writelines("Name, Time")
def attend_mark(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodelistknown = encoding(images)
print("encoding completes")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.2,0.2)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeface,faceloc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)
        matchIndex = np.argmin(facedis)
        # print(facedis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            print(name)
            x=face_recognition.face_locations(img)
            print(x)
            y1,x2,y2,x1 = x[0]
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x2,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            attend_mark(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
