from sklearn.neighbors import KNeighborsClassifier  # type: ignore
import cv2
import pickle
import numpy as np  # type: ignore
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch  # type: ignore

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the 'data' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Load previously saved labels and face data
with open('data/names.pk1', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pk1', 'rb') as f:
    FACES = pickle.load(f)

# Check and align the lengths of FACES and LABELS
print(f"Number of FACES samples: {len(FACES)}")
print(f"Number of LABELS samples: {len(LABELS)}")

min_samples = min(len(FACES), len(LABELS))
FACES = FACES[:min_samples]
LABELS = LABELS[:min_samples]

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgBackground = cv2.imread("background.png")
COL_NAMES = ['AADHAR NUMBER', 'NAME', 'VOTE', 'DATE', 'TIME']

# Dictionary to store vote counts
vote_counts = {
    "BJP": 0,
    "CONGRESS": 0,
    "JDS": 0,
    "NOTA": 0
}

# Capture the Aadhar number
aadhar_number = input("Enter your 12-digit Aadhar number: ")
while len(aadhar_number) != 12 or not aadhar_number.isdigit():
    print("Invalid Aadhar number. Please enter exactly 12 digits.")
    aadhar_number = input("Enter your 12-digit Aadhar number: ")

# Start the voting process
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Votes.csv")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        attendance = [aadhar_number, output[0], timestamp]

    imgBackground[300:300 + 480, 225:225 + 640] = frame
    cv2.imshow('frame', imgBackground)
    k = cv2.waitKey(1)

    def check_if_exists(value):
        try:
            with open("Votes.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0] == value:
                        return True
        except FileNotFoundError:
            print("File not found or unable to open the csv file")
        return False

    voter_exist = check_if_exists(aadhar_number)
    if voter_exist:
        print("YOU HAVE ALREADY VOTED")
        speak("YOU HAVE ALREADY VOTED")
        break

    if k == ord('1'):
        party = "BJP"
        speak("YOU HAVE VOTED FOR BJP")
    elif k == ord('2'):
        party = "CONGRESS"
        speak("YOU HAVE VOTED FOR CONGRESS")
    elif k == ord('3'):
        party = "JDS"
        speak("YOU HAVE VOTED FOR AAP")
    elif k == ord('4'):
        party = "NOTA"
        speak("YOU HAVE VOTED FOR NOTA")
    else:
        continue

    speak("YOUR VOTE HAS BEEN RECORDED")
    time.sleep(2)

    if exist:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            attendance = [aadhar_number, party, date, timestamp]
            writer.writerow(attendance)
    else:
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)
            attendance = [aadhar_number, party, date, timestamp]
            writer.writerow(attendance)

    vote_counts[party] += 1
    speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
    break

video.release()
cv2.destroyAllWindows()

# Display the vote counts
print("Vote Counts:")
for party, count in vote_counts.items():
    print(f"{party}: {count}")
