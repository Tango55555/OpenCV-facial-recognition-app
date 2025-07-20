# OpenCV-facial-recognition-app

This Python program uses face recognition to automate attendance tracking using a webcam. It detects known faces from a folder of images and logs the date and time into a CSV file when a known face is recognized.

## Features

- Real-time face detection using webcam
- Recognizes known faces from a given folder
- Logs attendance with timestamp to a CSV file
- Simple and lightweight implementation using `face_recognition` and OpenCV

## Requirements

- Python 3.6+
- Libraries:
  - `face_recognition`
  - `opencv-python`
  - `numpy`
  - `cv2`

## Instructions

basic.py shows off how the basic face recognition works by importing an image of a face and importing a different image of the same person and showing that the recognition realises that these are faces of the same person.
basic.py can be run using any python compiler / IDE

Attendance_project.py uses the face recognition and webcam to detect faces and track attendance. The ImagesAttendace folder stores all known faces and their names as the file names. When a face from the folder has been detected in the webcam, their name and the time of detection is written in attendance.csv. 
Attendance_project.py can be run using any python compiler / IDE
To add more faces into the known list, add an image of the person with the file name as their name into the ImagesAttendance folder.
