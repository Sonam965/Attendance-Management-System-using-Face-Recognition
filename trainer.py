import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Path for dataset
path = 'dataset'


# Prepare training data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        ids.append(Id)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)

    return np.array(ids), faces


ids, faces = getImagesAndLabels(path)
recognizer.train(faces, ids)

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
print(f"{len(np.unique(ids))} users trained successfully.")
