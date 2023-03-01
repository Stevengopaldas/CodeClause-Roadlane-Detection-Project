# CodeClause-Roadlane-Detection-Project

Road lane detection is an essential component of advanced driver assistance systems (ADAS) that helps in keeping vehicles on the correct path and avoiding accidents. The technology works by identifying the lane boundaries using image processing techniques and then alerting the driver in case the vehicle starts to drift from its designated lane.

The above code demonstrates how to implement road lane detection using OpenCV in Python. The code uses Canny edge detection and Hough transform to detect the lane lines in a given video stream. The process() function takes an image frame from the video stream, performs edge detection and applies a region of interest mask to extract the relevant portion of the image containing the lane lines. The Hough transform is then applied to identify the lines and draw them on the image.

The processed image frames are then written to a video file using the VideoWriter() function. The resulting output video can be used for further analysis and evaluation of the lane detection algorithm's performance. Overall, this code provides a basic implementation of road lane detection and can serve as a starting point for developing more advanced lane detection systems.
