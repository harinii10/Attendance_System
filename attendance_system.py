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

# names related to ids: example ==> Marcelo: id=1,  etc
# We will load these from names.txt if it exists, otherwise use a placeholder list
names = ['None'] 

if os.path.exists("names.txt"):
    with open("names.txt", "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                # Assuming IDs are sequential or we just append names. 
                # Since dataset_capture allows arbitrary IDs, this is a simple mapping approach.
                # Use a dictionary for better mapping in a real app, 
                # but for this list, we'll just append and try to match index if IDs are 1-based sequential.
                # BETTER APPROACH for random IDs: use a dict.
                pass 
else:
    print("names.txt not found. Names will be 'Unknown'.")

# Let's use a dictionary for ID -> Name mapping
id_names = {}
if os.path.exists("names.txt"):
    with open("names.txt", "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                id_names[int(parts[0])] = parts[1]

# Set to keep track of attendance marked in this session to avoid file I/O
marked_today = set()

# Load existing attendance for today to populate the set
if os.path.exists("attendance.csv"):
    with open("attendance.csv", "r") as f:
        reader = csv.reader(f)
        now_date = datetime.now().strftime('%Y-%m-%d')
        for row in reader:
            if len(row) >= 2 and row[1] == now_date:
                marked_today.add(row[0])

# Function to mark attendance
def mark_attendance(name):
    if name in marked_today:
        return

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    filename = "attendance.csv"
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
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
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less than 100 ==> "0" is perfect match 
        if (confidence < 100):
            name = id_names.get(id, "Unknown")
            confidence_text = "  {0}%".format(round(100 - confidence))
            
            # Additional check: Only mark attendance if we are reasonably sure (e.g., distance < 60)
            # This prevents false positives from marking attendance
            if confidence < 60 and name != "Unknown":
                mark_attendance(name)
        else:
            name = "Unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_text), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27 or k == ord('q'):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
