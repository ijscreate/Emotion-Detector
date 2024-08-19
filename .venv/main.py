import cv2
from deepface import DeepFace

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    try:
        #  Analyze the frame to detect the face and the dominant emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Handle list result
        if isinstance(result, list):
            result = result[0]

        # Get the bounding box coordinates
        face_coords = result['region']
        x, y, w, h = face_coords['x'], face_coords['y'], face_coords['w'], face_coords['h']

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Access the dominant emotion
        emotion = result['dominant_emotion']

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print("Error:", e)
        emotion = "No face detected"

    # Display the frame with the emotion label and bounding box
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
