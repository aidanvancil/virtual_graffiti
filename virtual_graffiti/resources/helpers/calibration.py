import cv2
import numpy as np
import os 
import json

def enumerate_cameras():
    """
    Enumerate the available camera indices.

    Returns:
        list: List of available camera indices.
    """
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        arr.append(index)
        cap.release()
        index += 1
    return arr
    
def on_trackbar(val):
    """
    Placeholder function for trackbar callback.

    Parameters:
        val: Trackbar value.

    Returns:
        None
    """
    pass

def find_color_ranges():
    """
    Find HSV color ranges for red, green, and purple lasers.

    Returns:
        tuple: Lower and upper HSV ranges for red, green, and purple lasers.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    colors = {}
    color_base = {
        'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
        'green': (np.array([50, 100, 100]), np.array([70, 255, 255])),
        'purple': (np.array([130, 50, 50]), np.array([160, 255, 255]))
    }   
    if os.path.exists("color_ranges.txt"):
        try:
            with open("color_ranges.txt", "r") as f:
                data = json.load(f)
                for color in data:
                    color_data = data[color]
                    lower = np.array(color_data['lower'])
                    upper = np.array(color_data['upper'])
                    color_base[color] = (lower, upper)
        except Exception as e:
            print(e)
             

    for color in ['red', 'green', 'purple']:
        print(f"Move the {color} laser into the frame and press Enter...")
        frame = None
        while True:
            _, frame = cap.read()
            cv2.imshow(f'{color.capitalize()} Laser Frame', frame)

            key = cv2.waitKey(1)
            if key == 13:
                break

        cv2.destroyAllWindows()

        lower = color_base[color][0]
        upper = color_base[color][1]
        colors[color] = (lower, upper)

        cv2.namedWindow(f'{color.capitalize()} HSV Sliders')
        print(lower, upper)
        cv2.createTrackbar('Hue Min', f'{color.capitalize()} HSV Sliders', lower[0], 180, on_trackbar)
        cv2.createTrackbar('Hue Max', f'{color.capitalize()} HSV Sliders', upper[0], 180, on_trackbar)
        cv2.createTrackbar('Saturation Min', f'{color.capitalize()} HSV Sliders', lower[1], 255, on_trackbar)
        cv2.createTrackbar('Saturation Max', f'{color.capitalize()} HSV Sliders', upper[1], 255, on_trackbar)
        cv2.createTrackbar('Value Min', f'{color.capitalize()} HSV Sliders', lower[2], 255, on_trackbar)
        cv2.createTrackbar('Value Max', f'{color.capitalize()} HSV Sliders', upper[2], 255, on_trackbar)

        while True:
            h_min = cv2.getTrackbarPos('Hue Min', f'{color.capitalize()} HSV Sliders')
            h_max = cv2.getTrackbarPos('Hue Max', f'{color.capitalize()} HSV Sliders')
            s_min = cv2.getTrackbarPos('Saturation Min', f'{color.capitalize()} HSV Sliders')
            s_max = cv2.getTrackbarPos('Saturation Max', f'{color.capitalize()} HSV Sliders')
            v_min = cv2.getTrackbarPos('Value Min', f'{color.capitalize()} HSV Sliders')
            v_max = cv2.getTrackbarPos('Value Max', f'{color.capitalize()} HSV Sliders')

            colors[color] = (np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))

            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), colors[color][0], colors[color][1])
            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow(f'{color.capitalize()} HSV Sliders', np.hstack([frame, result]))

            key = cv2.waitKey(1)
            if key == 13:  # If Enter key is pressed
                break

        cv2.destroyAllWindows()

    cap.release()
    return colors['red'][0], colors['red'][1], colors['green'][0], colors['green'][1], colors['purple'][0], colors['purple'][1]

def select_four_corners(image):    
    """
    Select four corners of an image.

    Parameters:
        image (numpy.ndarray): The image.

    Returns:
        list: List of four (x, y) coordinates representing the corners.
    """
    instructions = ['select top-right corner', 'select bottom-right corner', 'select bottom-left corner']

    minified_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    minified_image_initial = minified_image.copy()
    points = []
    num_points = 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal num_points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x * 2, y * 2))
            if not len(instructions):
                cv2.destroyAllWindows()
            else:
                text = instructions.pop(0)
                cv2.circle(minified_image, (x, y), 5, (0, 255, 0), -1)
                if num_points < 4:
                    minified_image_copy = minified_image.copy()
                    cv2.putText(minified_image_copy, text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.imshow("Select Corners", minified_image_copy)

    cv2.putText(minified_image_initial, 'select top-left corner', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow("Select Corners", minified_image_initial)
    cv2.setMouseCallback("Select Corners", mouse_callback)
    cv2.waitKey(0)
    return points

def get_skew_matrix(points, output_width, output_height):
    """
    Calculate the skew matrix for perspective transformation.

    Parameters:
        points (list): List of four (x, y) coordinates representing the corners.
        output_width (int): Width of the output image.
        output_height (int): Height of the output image.

    Returns:
        numpy.ndarray: Skew matrix for perspective transformation.
    """
    pts_src = np.array(points, dtype=np.float32)    
    pts_dst = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return matrix