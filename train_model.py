import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for i, imagePath in enumerate(imagePaths):
        # ignore system files like .DS_Store if any
        if os.path.split(imagePath)[-1].startswith("."):
            continue

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        # extract the face ID from the image name
        # Format: User.ID.SampleNum.jpg
        try:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        except (IndexError, ValueError):
            print(f"[WARNING] Skipping invalid file format: {imagePath}")
            continue

        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
            
        # Print progress
        if i % 50 == 0 and i > 0:
            print(f" [INFO] Processed {i}/{len(imagePaths)} images...")

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)

if len(faces) == 0:
    print("[ERROR] No face images found in 'dataset/'. Please run 'dataset_capture.py' first.")
    exit()

recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
if not os.path.exists('trainer'):
    os.makedirs('trainer')

recognizer.write('trainer/trainer.yml') 

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
