import cv2
import numpy as np

def enumerate_cameras():
    index = 0
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        cap.release()
        index += 1

    return index

def color_segmentation(frame, lower_color, upper_color):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only specified color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Bitwise-AND mask and original image
    segmented = cv2.bitwise_and(frame, frame, mask=mask)

    return segmented

def edge_detection(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def update_canvas_color(canvas, x, y, color):
    # Draw a colored point on the canvas
    cv2.circle(canvas, (x, y), 3, color, -1)

def get_color_from_index(index):
    # Define colors for each laser
    colors = [(0, 0, 255), (0, 255, 0)]  # Red and Green

    # Return the corresponding color for the given index
    return colors[index]

if __name__ == "__main__":
    camera_index = enumerate_cameras()

    if camera_index == 0:
        print("No cameras found.")
    else:
        print(f"Number of available cameras: {camera_index}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Define color ranges for red and green lasers (adjust these based on your laser colors)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])

        # Set up canvas window
        canvas_width, canvas_height = 800, 600
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas_window_name = 'Canvas'
        cv2.namedWindow(canvas_window_name)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Apply color segmentation for red laser
            red_segmented = color_segmentation(frame, lower_red, upper_red)

            # Apply color segmentation for green laser
            green_segmented = color_segmentation(frame, lower_green, upper_green)

            # Update canvas based on laser positions and color
            for i, segmented in enumerate([red_segmented, green_segmented]):
                edges = edge_detection(segmented)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 50:  # Adjust the area threshold as needed
                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            color = get_color_from_index(i)
                            update_canvas_color(canvas, cx, cy, color)

            # Display the original frame, red laser edges, green laser edges, and canvas
            cv2.imshow('Original', frame)
            cv2.imshow('Red Laser Edges', edge_detection(red_segmented))
            cv2.imshow('Green Laser Edges', edge_detection(green_segmented))
            cv2.imshow(canvas_window_name, canvas)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
