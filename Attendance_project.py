import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime


# Function that takes in a list of images and returns a list of encodings for the images
def find_encodings(image_list):
    encode_list = []
    for image in image_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)
    return encode_list


# Function that marks the name and time of detection in attendance.csv
def mark_attendance(attendance_name):
    with open('attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list =[]
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if attendance_name not in line:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines("\n" + attendance_name + ", " + dt_string)


path = 'ImagesAttendance'
images = []
className = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
for x, cl in enumerate(myList):
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])

encodeListKnown = find_encodings(images)
print('Encodings Complete')
print('Press q to quit webcam')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize image to become smaller to make program run faster
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find webcam faces and their encodings
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Match found face encodings on webcam to known list
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Match is the one with the lowest distance
        matchIndex = np.argmin(faceDis)

        # Labels name as unknown if face doesn't match any of the encodings
        if faceDis[matchIndex] < 0.50:
            name = className[matchIndex].upper()
            mark_attendance(name)
        else:
            name = 'Unknown'

        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()