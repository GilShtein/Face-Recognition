# Face-Recognition
This Python project utilizes OpenCV to perform face recognition by comparing histograms of detected faces with a known face. The similarity between histograms is computed using correlation metrics.

## Features
- __Histogram Calculation__: Compute histograms of images to represent their color distribution.
- __Face Detection__: Utilize Haar cascades to detect faces in live camera streams.
- __Histogram Comparison__: Compare histograms of detected faces with a known face to determine similarity.
- __Real-time Recognition__: Perform face recognition in real-time using live camera feed.
- __Thresholding__: Determine if a detected face is known or unknown based on a predefined similarity threshold.
## Requirements
Python 3.x
OpenCV (cv2)
Matplotlib (for histogram visualization)
