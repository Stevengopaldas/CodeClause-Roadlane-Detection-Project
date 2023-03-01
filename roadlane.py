# I have given two ways (one in comments). is another way of detecting the lanes
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt


# def roi(image, vertices):
#     mask = np.zeros_like(image)
#     mask_color = 260
#     cv2.fillPoly(mask, vertices, mask_color)
#     cropped_img = cv2.bitwise_and(image, mask)
#     return cropped_img


# def draw_lines(image, hough_lines):
#     for line in hough_lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(image, (x1, y1), (x2, y2), (0, 260, 0), 2)

#     return image

# def process(img):
#     height = img.shape[0]
#     width = img.shape[1]
#     roi_vertices = [
#         (0, 680),
#         (2*width/3, 2*height/3),
#         (width, 1020)
#     ]

#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))

#     canny = cv2.Canny(gray_img, 130, 220)

#     roi_img = roi(canny, np.array([roi_vertices], np.int32))

#     lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, threshold=15, minLineLength=15, maxLineGap=2)

#     final_img = draw_lines(img, lines)

#     return final_img


# cap = cv2.VideoCapture("C:/Users/Steven/Downloads/roadlane0.mp4")

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*"XVID")

# saved_frame = cv2.VideoWriter("lane_detection.avi", fourcc, 30.0, (frame_width, frame_height))

# while cap.isOpened():
#     ret, frame = cap.read()

#     try:
#         frame = process(frame)

#         saved_frame.write(frame)
#         cv2.imshow("frame", frame)

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     except Exception:
#         break

# cap.release()

# saved_frame.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np

def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    return image

def process(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, threshold1=50, threshold2=150)
    cropped_image = roi(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=20, lines=np.array([]), minLineLength=10, maxLineGap=2)

    line_image = np.zeros_like(image)
    if lines is not None:
        line_image = draw_lines(line_image, lines)

    return cv2.addWeighted(image, 0.8, line_image, 1, 0)

cap = cv2.VideoCapture("C:/Users/Steven/Downloads/roadlane0.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    try:
        frame = process(frame)

        cv2.imshow("Road Lane Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception:
        break

cap.release()
cv2.destroyAllWindows()

