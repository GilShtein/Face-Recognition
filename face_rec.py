import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Function to compute histogram of an image
def compute_histogram(image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Calculate histogram
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist = cv.normalize(hist, hist).flatten()
    return hist


# Load known image and compute its histogram
known_image = cv.imread("my_face.png")

known_histogram = compute_histogram(known_image)




# Launch the live camera
cam = cv.VideoCapture(0)
# Check camera
if not cam.isOpened():
    print("Camera not working")
    exit()

# Load Haar cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# When camera is opened
while True:
    # Capture the image frame-by-frame
    ret, frame = cam.read()

    # Check if frame is reading or not
    if not ret:
        print("Can't receive the frame")
        break

    # Face detection in the frame
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_locations = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for (x, y, w, h) in face_locations:
        # Draw a rectangle around detected faces
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Get the region of the detected face
        detected_face = frame[y:y + h, x:x + w]

        # Compute histogram for detected face
        detected_histogram = compute_histogram(detected_face)

        # Compare the histogram with the known face
        # Using correlation metric (1.0 is identical, -1.0 is completely different)
        similarity = cv.compareHist(known_histogram, detected_histogram, cv.HISTCMP_CORREL)



        print(similarity)
        if similarity > 0.7 :  # Threshold for similarity
            frame = cv.putText(frame, 'known', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        else:
            frame = cv.putText(frame, 'Unknown Person', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Frame', frame)

    # End the streaming
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture
cam.release()
cv.destroyAllWindows()
