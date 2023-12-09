import cv2
from keras.models import load_model
import numpy as np

emotion_model = load_model('assets/models/model_v6_23.hdf5', compile=False)
emotions = ['Sad', 'Neutral', 'Happy', 'Sad', 'Happy', 'Sad', 'Neutral']
threshold = 0.2
vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('assets/models/haarcascade_frontalface.xml')

while 1:
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 9)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        emotion_prediction = emotion_model.predict(roi_gray)

        max_index = np.argmax(emotion_prediction)
        max_emotion = emotions[max_index]

        cv2.putText(frame, f'Emotion: {max_emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
