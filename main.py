import cv2 as cv 
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

# Initialize video capture from the default camera (0)
cap = cv.VideoCapture(0)

# Set video frame dimensions to 640x480 pixels
cap.set(3, 640)
cap.set(4, 480)

# Create a ColorFinder object
myColorFinder = ColorFinder(False)

# Define initial HSV color range values
hsvVals = {'hmin': 23, 'smin': 0, 'vmin': 195, 'hmax': 179, 'smax': 255, 'vmax': 255}

# Function for the trackbar callback (does nothing)
def empty(a):
    pass

# Create a settings window with trackbars for threshold adjustment
cv.namedWindow("Settings")
cv.resizeWindow("Settings", 640, 240)
cv.createTrackbar("Threshold1", "Settings", 27, 255, empty)
cv.createTrackbar("Threshold2", "Settings", 120, 255, empty)

# Function for image preprocessing
def preProcessing(img):
    # Apply Gaussian blur to reduce noise
    imgPre = cv.GaussianBlur(img, (5, 5), 3)
    
    # Retrieve threshold values from trackbars
    thresh1 = cv.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv.getTrackbarPos("Threshold2", "Settings")
    
    # Apply Canny edge detection
    imgPre = cv.Canny(imgPre, thresh1, thresh2)
    
    # Perform dilation and morphological closing
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv.dilate(imgPre, kernel, iterations=1)
    imgPre = cv.morphologyEx(imgPre, cv.MORPH_CLOSE, kernel)
    
    return imgPre

# Main loop for video processing
while True:
    # Capture a frame from the camera
    _, image = cap.read()
    
    # Preprocess the frame
    imgPre = preProcessing(image)
    
    # Find contours in the processed frame
    imgContours, conFound = cvzone.findContours(image, imgPre, minArea=20)
    
    # Initialize a variable to keep track of the total money
    money = 0
    
    # Create an empty image for displaying the money count
    imgCount = np.zeros((480, 640, 3), np.uint8)
    
    # Check if any contours are found
    if conFound:
        for count, contour in enumerate(conFound):
            peri = cv.arcLength(contour["cnt"], True)
            approx = cv.approxPolyDP(contour["cnt"], 0.02 * peri, True)
            
            # Check if the contour has more than 5 corners (an arbitrary threshold)
            if len(approx) > 5:
                area = contour["area"]
                x, y, w, h = contour['bbox']
                imgCrop = image[y:y+h, x:x+w]
                
                # Update color information and mask for the cropped region
                imgColor, mask = myColorFinder.update(imgCrop, hsvVals)
                whitepixel = cv.countNonZero(mask)
                
                # Determine the monetary value based on the area
                if area < 4500:
                    money += 5
                elif 4500 < area < 5850:
                    money += 10
                elif 5890 < area < 7890:
                    money += 20
                elif area > 8000:
                    money += 40
    
    # Display the total money count on the image
    cvzone.putTextRect(imgCount, f'ksh.{money}', (100, 300), scale=10, colorR=(0, 0, 255), thickness=5)   
    
    # Stack the original image, preprocessed image, contours, and money count image for display
    imageStacked = cvzone.stackImages([image, imgPre, imgContours, imgCount], 2, 0.5)
    
    # Display the stacked image with the money count
    cvzone.putTextRect(imageStacked, f'ksh.{money}', (50, 50), colorR=(0, 0, 255))
    cv.imshow("Money Counter", imageStacked)
    
    # Wait for a key press and check if 'q' is pressed to exit the loop
    key = cv.waitKey(1)
    if key == ord("q"):
        break

# Release the video capture and close all windows
cap.release() 
cv.destroyAllWindows()
