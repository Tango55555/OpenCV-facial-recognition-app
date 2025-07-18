import face_recognition
import cv2
import numpy as np
import os


# Function that takes in a list of images and returns a list of encodings for the images
def find_encodings(image_list):
    encode_list = []
    for image in image_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)
    return encode_list


path = 'ImagesAttendance'
images = []
className = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for x, cl in enumerate(myList):
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])

encodeListKnown = find_encodings(images)
print('Encodings Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Resize image to become smaller to make program run faster
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find webcam faces and their encodings
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Match found face encodings to known list
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Match is the one with the lowest distance
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc

            # Multiplied by 4 because image was scaled down prior
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)