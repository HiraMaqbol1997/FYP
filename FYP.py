from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv

path = get_file("gender_detection.model","/home/hp/PycharmProjects/gender-detection-keras-master", cache_subdir="trained_Model", cache_dir=os.getcwd())

mdl = load_model(path)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error in opening camera")
    exit()
    
categories = ['male','female']

while camera.isOpened():

    check, frame = camera.read()

    if not check:
        print("Error in Reading Frames")
        exit()

    face, value = cv.detect_face(frame)
    print(face)
    print(value)


    for idx, arr in enumerate(face):

        (start_X, start_Y) = arr[0], arr[1]
        (end_X, end_Y) = arr[2], arr[3]

        cv2.rectangle(frame, (start_X,start_Y), (end_X,end_Y), (200, 255, 200), 3)

        face_rec = np.copy(frame[start_Y:end_Y,start_X:end_X])

        if (face_rec.shape[0]) < 10 or (face_rec.shape[1]) < 10:
            continue


        face_rec = cv2.resize(face_rec, (96,96))
        face_rec = face_rec.astype("float") / 255.0
        face_rec = img_to_array(face_rec)
        face_rec = np.expand_dims(face_rec, axis=0)


        conf = mdl.predict(face_rec)[0]
        print(conf)
        print(categories)


        idx = np.argmax(conf)
        label = categories[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = start_Y - 10 if start_Y - 10 > 10 else start_Y + 10


        cv2.putText(frame, label, (start_X, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 3)


    cv2.imshow("gender detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
