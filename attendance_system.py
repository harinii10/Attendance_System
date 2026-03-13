import cv2
import numpy as np
import os
import csv
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists('trainer/trainer.yml'):
    print("[ERROR] 'trainer/trainer.yml' not found. Please run 'train_model.py' first.")
    exit()

recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# initiatize id counter
id = 0

# Let's use a dictionary for ID -> Name mapping
id_names = {}
if os.path.exists("names.csv"):
    with open("names.csv", "r") as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip header
        except StopIteration:
            pass
        for row in reader:
            if len(row) >= 2:
                try:
                    id_names[int(row[0])] = row[1]
                except ValueError:
                    pass
else:
    print("names.csv not found. Names will be 'Unknown'.")

# Set to keep track of attendance marked in this session to avoid file I/O
marked_today = set()

# Get today's attendance filename
now_date = datetime.now().strftime('%Y-%m-%d')
attendance_filename = f"Attendance_{now_date}.csv"

# Load existing attendance for today to populate the set
if os.path.exists(attendance_filename):
    with open(attendance_filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 1 and row[0] != 'Name':
                marked_today.add(row[0])

# Function to mark attendance
def mark_attendance(name):
    if name in marked_today:
        return

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    file_exists = os.path.exists(attendance_filename)
    
    with open(attendance_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time'])
        writer.writerow([name, date_str, time_str])
    
    marked_today.add(name)
    print(f"Attendance marked for {name}")

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

print("\n [INFO] Starting Attendance System. Press 'q' to exit.")

while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically if needed
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less than 100 ==> "0" is perfect match 
        if (confidence < 100):
            # Additional check: Only mark attendance if we are reasonably sure (e.g., distance < 60)
            if confidence < 60:
                name = id_names.get(id, "Unknown")
                confidence_text = "  {0}%".format(round(100 - confidence))
                if name != "Unknown":
                    mark_attendance(name)
                    color = (0, 255, 0) # Green for recognized
                else:
                    color = (0, 0, 255) # Red for unknown ID
            else:
                name = "Unknown"
                confidence_text = "  {0}%".format(round(100 - confidence))
                color = (0, 0, 255) # Red for uncertain
        else:
            name = "Unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))
            color = (0, 0, 255) # Red for unknown
            
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        
        # Display name and confidence
        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_text), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('Attendance System',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27 or k == ord('q'):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
