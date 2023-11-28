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


            # Display the original frame, red laser edges, green laser edges, and canvas
            cv2.imshow('Original', frame)
            cv2.imshow(canvas_window_name, canvas)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
