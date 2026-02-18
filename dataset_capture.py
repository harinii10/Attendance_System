import cv2
import os

# Create dataset directory if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load the Haar Cascade for face detection
if not os.path.exists('haarcascade_frontalface_default.xml'):
    print("[ERROR] 'haarcascade_frontalface_default.xml' not found. Please download it or check the path.")
    exit()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Get user input
face_id = input('\n enter user id end press <return> ==>  ')
face_name = input('\n enter user name end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
# Initialize individual sampling face count
count = 0

if not cam.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Camera opened. Starting loop...")

while(True):
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break
        
    # Flip the image vertically (optional, depending on camera mounting)
    # img = cv2.flip(img, -1) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        # Format: User.ID.SampleNum.jpg
        # We also save the name in a separate file or just keep track manually for this simple project
        # Ideally, we'd map ID to Name in the recognition script.
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

# Create/Append to a mapping file for ID to Name
with open("names.txt", "a") as f:
    f.write(f"{face_id},{face_name}\n")
