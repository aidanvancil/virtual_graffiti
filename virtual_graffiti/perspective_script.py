import cv2 as cv
import numpy as np
import logging

# pts_src is an array of points (x, y) picked by the user
pts_src = []
coords_picked = 0

# function that openCV uses when a left mouse click occurs
# also adds points to pts_src array
# also puts a yellow dot where the user selected the point
def click_coord(event, x, y, flags, params):
        global pts_src
        if event == cv.EVENT_LBUTTONDOWN:
                pts_src.append([x, y])
                cv.circle(frame, (x,y), 3, (0,255,255), -1)

# valid color ranges for tracking
color_ranges = {
        'red': ([0, 100, 100], [5, 255, 255]),
        'purple': ([130, 50, 50], [160, 255, 255]),
        'green': ([50, 50, 50], [80, 255, 255]),
    }

# returns value of specific colors
def get_color(color):
        if color == 'red':
            return (0, 0, 255)
        elif color == 'purple':
            return (255, 0, 255)
        elif color == 'green':
            return (0, 255, 0)
        else:
            return (255, 255, 255)

# video capturing via openCV
cap = cv.VideoCapture(-1)
cap.set(cv.CAP_PROP_FPS, 60)

# this is an early test for logging to get an idea of how it would work
if not cap.isOpened():
        logging.critical('Camera failed to open')
        exit()

# main function loop
while True:
        ret, frame = cap.read()
        if not ret:
                logging.critical('Frame from camera stream failed')
                break
        while not coords_picked:
                cv.namedWindow('Pick Coordinates')
                cv.setMouseCallback('Pick Coordinates', click_coord)
                cv.imshow('Pick Coordinates', frame)
                cv.waitKey(1)

                # this requires the user to input four points to be used in pts_src
                if len(pts_src) == 4:
                        coords_picked = 1
                        cv.destroyAllWindows()

        else:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                color_centers = {'red': [], 'purple': [], 'green': []}

                for color, (lower, upper) in color_ranges.items():
                        mask = cv.inRange(hsv, np.array(lower), np.array(upper))

                        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        largest_contour = max(contours, key=cv.contourArea, default=None)

                        if largest_contour is not None:
                                M = cv.moments(largest_contour)
                                if M["m00"] != 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        color_centers[color] = (cx, cy)

                for color, center in color_centers.items():
                        if center:
                                cv.circle(frame, center, 10, get_color(color), -1)

                # the reason we change it to float32 is because the getPersepective
                # function in openCV expects that to be the form of the array
                pts_src = np.float32(pts_src)

                # this destination matrix can be changed based on our desired parameter etc
                pts_dst = np.float32([[0,0], [320,0], [0,240], [320,240]])

                # getPerspectiveTransform returns a matrix that we then use in the warpPerspective
                M = cv.getPerspectiveTransform(pts_src, pts_dst)
                frame_out = cv.warpPerspective(frame, M, (320, 240))
                cv.imshow('warped', frame_out)
                if cv.waitKey(1) == ord('q'):
                        break

cap.release()
cv.destroyAllWindows()

