import cv2
import sqlite3
from datetime import datetime
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

# Ensure dataset directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')


# Create table if not exists
def create_table():
    conn = sqlite3.connect("students.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS STUDENTS (
                    ID INTEGER PRIMARY KEY,
                    NAME TEXT NOT NULL,
                    AGE INTEGER NOT NULL,
                    REG_DATE TEXT
                );''')
    conn.commit()
    conn.close()


# Insert or update user details
def insertorupdate(Id, Name, Age):
    conn = sqlite3.connect("students.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE ID=?", (Id,))

    if cursor.fetchone():
        conn.execute("UPDATE STUDENTS SET NAME=?, AGE=? WHERE ID=?", (Name, Age, Id))
    else:
        conn.execute("INSERT INTO STUDENTS (ID, NAME, AGE, REG_DATE) VALUES (?, ?, ?, ?)",
                     (Id, Name, Age, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()


create_table()

Id = input('Enter User Id: ')
Name = input('Enter User Name: ')
Age = input('Enter User Age: ')

insertorupdate(Id, Name, Age)
print("Initial user details added. Now capturing face data...")

sampleNum = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        cv2.imwrite(f"dataset/user.{Id}.{sampleNum}.jpg", face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)

    cv2.imshow("Face", img)
    cv2.waitKey(1)

    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
print("Face data collected successfully.")
