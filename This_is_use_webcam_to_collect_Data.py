import cv2
import numpy as np
import os

classifier = cv2.CascadeClassifier(
    r"C:/Users/subhadip sinha/Desktop/Face detection/haarcascade_frontalface_default (2).xml"
)

# Initialize webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

# Create 'images' folder if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

data = []

while len(data) < 100:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    face_points = classifier.detectMultiScale(frame, 1.3, 5)

    if len(face_points) > 0:
        for x, y, w, h in face_points:
            face_frame = frame[y : y + h + 1, x : x + w + 1]
            cv2.imshow("Only face", face_frame)
            if len(data) < 100:
                print(len(data) + 1, "/100")
                data.append(face_frame)
                break  # Only take one face per frame

    cv2.putText(
        frame, str(len(data)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
    )
    cv2.imshow("frame", frame)

    if cv2.waitKey(30) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(data) == 100:
    name = input("Enter Face holder name : ")
    for i in range(100):
        cv2.imwrite(f"images/{name}_{i}.jpg", data[i])
    print("Done")
else:
    print("Need more data")
