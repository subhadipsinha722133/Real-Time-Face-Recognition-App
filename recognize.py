# import urllib
# import cv2
# import numpy as np
# from keras.models import load_model

# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# model = load_model("final_model.h5")

# URL = 'http://192.168.33.78:8080/shot.jpg'

# def get_pred_label(pred):
#     labels = ["akash","chandru","rakesh","shiva","siddhant"]
#     return labels[pred]

# def preprocess(img):
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img,(100,100))
#     img = cv2.equalizeHist(img)
#     img = img.reshape(1,100,100,1)
#     img = img/255
#     return img


# ret = True
# while ret:

#     img_url = urllib.request.urlopen(URL)
#     image = np.array(bytearray(img_url.read()),np.uint8)
#     frame = cv2.imdecode(image,-1)

#     faces = classifier.detectMultiScale(frame,1.5,5)

#     for x,y,w,h in faces:
#         face = frame[y:y+h,x:x+w]
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
#         cv2.putText(frame,get_pred_label(np.argmax(model.predict(preprocess(face)))),
#                     (200,200),cv2.FONT_HERSHEY_COMPLEX,1,
#                     (255,0,0),2)

#     cv2.imshow("capture",frame)
#     if cv2.waitKey(1)==ord('q'):
#         break

# cv2.destroyAllWindows()

import cv2
import numpy as np
from keras.models import load_model

classifier = cv2.CascadeClassifier(
    r"C:\Users\subhadip sinha\Desktop\Face detection\haarcascade_frontalface_default (2).xml"
)
model = load_model(r"C:\Users\subhadip sinha\Desktop\Face detection\final_model.h5")


def get_pred_label(pred):
    labels = ["Hrishab", "Hrithik", "Saurav", "Subhadip", "Vishajit"]
    return labels[pred]


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img


# Initialize webcam (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    faces = classifier.detectMultiScale(frame, 1.5, 5)

    for x, y, w, h in faces:
        face = frame[y : y + h, x : x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # Predict label
        pred = model.predict(preprocess(face))
        label = get_pred_label(np.argmax(pred))

        cv2.putText(
            frame, label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2
        )

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
