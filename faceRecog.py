import cv2
from deepface import DeepFace

# Load the pre-trained face cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open a video capture object (try default camera first)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Cannot open camera 0, trying camera 1...")
    cap = cv2.VideoCapture(1)  # Try alternate camera

# If still not opened, handle the error gracefully
if not cap.isOpened():
    print("Error: Cannot open webcam. Please check camera index or permissions.")
else:
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Analyze the frame for emotion using DeepFace
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            results = result[0]

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the dominant emotion text on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, results['dominant_emotion'], (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the original video frame
            cv2.imshow('Original video', frame)

        except Exception as e:
            print(f"Error during analysis: {e}")

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
