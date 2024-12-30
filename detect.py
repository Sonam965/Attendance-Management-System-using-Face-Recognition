import cv2
import sqlite3
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load trained model
recognizer.read('recognizer/trainingdata.yml')


# Fetch user details from DB
def getProfile(Id):
    conn = sqlite3.connect("students.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE ID=?", (Id,))
    profile = cursor.fetchone()  # Fetch one user directly
    conn.close()
    return profile


cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    detected = False  # Track if any face is detected

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        Id, confidence = recognizer.predict(face)

        if confidence < 70:  # Recognition threshold
            profile = getProfile(Id)
            if profile:
                detected = True
                user_info = f"ID: {profile[0]}, Name: {profile[1]}, Age: {profile[2]}"
                print(user_info)  # Print user info to terminal

                # Display on camera window
                cv2.putText(img, user_info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Recognizing Face", img)

    # Close window automatically after detection
    if detected:
        print("Face detected. Closing camera...")
        cv2.waitKey(2000)  # Keep window for 2 seconds
        break

    # Press 'q' to manually quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Camera closed.")
