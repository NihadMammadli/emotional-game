import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_model = cv2.dnn.readNetFromTensorflow('emotion_detector_models/emotion_detector_model.pb')

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to detect and classify emotions
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face region to match the input size of the emotion model
        face_roi = cv2.resize(face_roi, (48, 48))

        # Preprocess the image for the emotion model
        face_roi = face_roi.reshape(1, 48, 48, 1)
        face_roi = face_roi / 255.0

        # Make predictions using the emotion model
        emotion_model.setInput(face_roi)
        predictions = emotion_model.forward()

        # Get the index of the predicted emotion
        emotion_index = predictions[0].argmax()

        # Display the emotion text on the frame
        emotion_text = emotions[emotion_index]
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

# Open a video capture object (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform emotion detection
    frame = detect_emotion(frame)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
