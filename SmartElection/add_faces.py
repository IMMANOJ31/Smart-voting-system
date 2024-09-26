import cv2  # type: ignore
import pickle
import numpy as np  # type: ignore
import os

# Ensure the 'data' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_valid_aadhar():
    while True:
        aadhar_number = input("Enter your 12-digit Aadhar number: ")
        if len(aadhar_number) == 12 and aadhar_number.isdigit():
            return aadhar_number
        else:
            print("Invalid Aadhar number. Please enter exactly 12 digits.")

name = get_valid_aadhar()
faces_data = []
framesTotal = 65
captureAfterFrame = 2
i = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= framesTotal and i % captureAfterFrame == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

# Release video capture and destroy all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Process the face data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((framesTotal, -1))
print(faces_data)

# Save face data and names using pickle
if 'names.pk1' not in os.listdir('data/'):
    names = [name] * framesTotal
    with open('data/names.pk1', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pk1', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * framesTotal
    with open('data/names.pk1', 'wb') as f:
        pickle.dump(names, f)

if 'faces.pk1' not in os.listdir('data/'):
    with open('data/faces_data.pk1', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pk1', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pk1', 'wb') as f:
        pickle.dump(faces, f)
