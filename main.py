import cv2
from deepface import DeepFace

# This line creates a video capture object. 0 means it will use the default webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# --- NEW CODE FOR OPTIMIZATION ---

# We need a counter to keep track of the frames as they come in.
frame_counter = 0 
# We will only perform the heavy analysis on every 5th frame.
# You can change this number. A higher number means smoother video but less frequent updates.
FRAME_ANALYSIS_INTERVAL = 5
# This variable will hold the results from the last successful analysis.
last_results = []

# --- END OF NEW CODE ---

# This is the main loop that will run continuously
while True:
    # Read a new frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Increment our frame counter
    frame_counter += 1

    # --- MODIFIED LOGIC ---
    # Only run the heavy DeepFace analysis if it's the 5th, 10th, 15th frame, etc.
    if frame_counter % FRAME_ANALYSIS_INTERVAL == 0:
        try:
            # We run the analysis on the current frame.
            # The results are stored in our 'last_results' variable, overwriting the old ones.
            last_results = DeepFace.analyze(
                frame,
                actions=['gender', 'emotion'],
                enforce_detection=False,      # Don't crash if no face is found
                detector_backend='mtcnn'      # Use the more accurate face detector
            )
        except Exception as e:
            # If any error happens during analysis (like no face found),
            # we clear the results to avoid drawing old boxes.
            last_results = []

    # For EVERY frame (1, 2, 3, 4, 5, 6...), we draw the results from the LAST successful analysis.
    # This makes the video look smooth.
    if last_results:
        for result in last_results:
            # Get the bounding box for the face
            box = result['region']
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            
            # Draw the green rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get the gender and emotion
            gender = result['dominant_gender']
            emotion = result['dominant_emotion']

            # Create the text to display
            text = f"{gender} - {emotion}"
            
            # Put the text above the rectangle
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # --- END OF MODIFIED LOGIC ---

    # Display the final frame in a window
    cv2.imshow('Guardian Camcorder', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When the loop is broken, release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

