from django.http import JsonResponse
from screeninfo import get_monitors
import random
import cv2
import threading
import numpy as np
from queue import Queue
import math
import os 
import time
from datetime import datetime, timedelta
import socket
import threading

SKEWED = False
selected_points = None
matrix = None

def on_trackbar(val):
    pass

def find_color_ranges():
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
                lines = f.readlines()
                red_lower = eval(lines[0].split(":")[1].strip())
                red_upper = eval(lines[1].split(":")[1].strip())
                green_lower = eval(lines[2].split(":")[1].strip())
                green_upper = eval(lines[3].split(":")[1].strip())
                purple_lower = eval(lines[4].split(":")[1].strip())
                purple_upper = eval(lines[5].split(":")[1].strip())
            color_base = {
                'red': (red_lower, red_upper),
                'green': (green_lower, green_upper),
                'purple': (purple_lower, purple_upper)
            }
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
        colors[color] = (lower, upper )

        cv2.namedWindow(f'{color.capitalize()} HSV Sliders')

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
    global SKEWED
    global selected_points
    
    if SKEWED:
        return selected_points

    instructions = ['select top-right corner', 'select bottom-right corner', 'select bottom-left corner']

    minified_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    minified_image_initial = minified_image.copy()
    points = []
    num_points = 0
    
    # Mouse callback function to get user clicks
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

    # Initial display of instructions
    cv2.putText(minified_image_initial, 'select top-left corner', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow("Select Corners", minified_image_initial)
    cv2.setMouseCallback("Select Corners", mouse_callback)
    cv2.waitKey(0)
    
    selected_points = points
    SKEWED = True
    
    return points



def get_skew_matrix(points, output_width, output_height):
    global matrix
    pts_src = np.array(points, dtype=np.float32)    
    pts_dst = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

def handle_client_connection(conn, image_queue, command_queue):
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        if data == 'fill':
            command_queue.put(data)
        elif data == 'party':
            command_queue.put(data)
        else:
            image_queue.put(data)

def enumerate_cameras():
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

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

#NFR6
def is_laser_contour(contour, hsv_frame, min_area=20, max_area=200):
    area = cv2.contourArea(contour)
    return min_area < area < max_area

def color_segmentation(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def load_scaled_image(image_path, width, height):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
    image_path = os.path.join(base_dir, image_path.lstrip('/'))
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not found")
    return cv2.resize(image, (width, height))

#UC04
def update_canvas_with_image(canvas, background_image, x, y, scale_factor, radius=180):
    scaled_x = int(x * scale_factor)
    scaled_y = int(y * scale_factor)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)
    mask_indices = np.where(mask > 0)
    canvas[mask_indices] = background_image[mask_indices]

# def is_registered(color):
#     red = (0, 0, 255)
#     green = (0, 255, 0)
#     color_map = {
#         str(green): (None, None),
#         str(red): (None, None)
#     }
#     try:
#         laser = Laser.objects.get(color=color_map[color])
#     except Laser.DoesNotExist:
#         return False

#     try:
#         UserProfile.objects.get(laser=laser)
#         return True
#     except UserProfile.DoesNotExist:
#         return False

def smooth_drawing(last_point, current_point, canvas, color=(0, 0, 255), thickness=2, distance_threshold=160):
    if last_point is not None and calculate_distance(last_point, current_point) < distance_threshold: # and is_registered(color):
        cv2.line(canvas, last_point, current_point, color, thickness, lineType=cv2.LINE_AA)
    return current_point

def count_filled_pixels(canvas, background_image):
    filled_pixels = np.sum(np.any(canvas != [0, 0, 0], axis=2))
    total_pixels = background_image.shape[0] * background_image.shape[1]
    return filled_pixels, total_pixels

#UC11
def find_black_coordinates(canvas):
    black_coords = np.argwhere(np.all(canvas == [0, 0, 0], axis=-1))
    return black_coords

def apply_glitter_effect(canvas, background_image, iterations=400, intensity=10000, delay=2):
    black_coords = find_black_coordinates(canvas)
    for _ in range(iterations):
        for _ in range(intensity):
            index = random.randint(0, len(black_coords) - 1)
            y, x = black_coords[index]
            canvas[y, x] = background_image[y, x]
        cv2.imshow('Canvas', canvas)
        cv2.waitKey(delay)


def skew_point(point, skew_matrix):
    homogeneous_point = np.array([point[0], point[1], 1])
    skewed_point = np.dot(skew_matrix, homogeneous_point)
    return (int(skewed_point[0] / skewed_point[2]), int(skewed_point[1] / skewed_point[2]))

def clear_canvas(canvas):
    canvas[:, :] = 0

def get_clear_button(skew_matrix, clear_area_rect):
    # Skew the clear button position
    skewed_clear_area_rect = [skew_point(clear_area_rect[0], skew_matrix),
                              skew_point(clear_area_rect[1], skew_matrix),
                              skew_point(clear_area_rect[3], skew_matrix),
                              skew_point(clear_area_rect[2], skew_matrix),
                            ]
    return skewed_clear_area_rect

def get_color_palette(skew_matrix, color_pallete_rect):
    skewed_clear_area_rect = [skew_point(color_pallete_rect[0], skew_matrix),
                              skew_point(color_pallete_rect[1], skew_matrix),
                              skew_point(color_pallete_rect[3], skew_matrix),
                              skew_point(color_pallete_rect[2], skew_matrix),
                            ]
    return skewed_clear_area_rect

def draw_palette_box(skewed_canvas, palette_box, color):
    list_ver = [list(i) for i in palette_box]
    pts = np.array(list_ver, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(skewed_canvas, [pts], color)
    return skewed_canvas

def draw_clear_button(skewed_canvas, skewed_clear_area_rect):
    # Convert coordinates to integer and reshape as needed
    # Draw the skewed clear button rectangle

    list_ver = [list(i) for i in skewed_clear_area_rect]
    pts = np.array(list_ver, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(skewed_canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    return skewed_canvas

#FR1, UC10
def init():
    global SKEWED
    global matrix
    global selected_points

    mode_status = 'offline' #change eventually
    mode = 'free'
    image_queue = Queue()
    command_queue = Queue()
    # camera_indexes = enumerate_cameras()
    # print(camera_indexes)
    # if len(camera_indexes) == 0:
    #     print('No cameras found')
    #     return
    # print('Camera found')

    red_lower, red_upper, green_lower, green_upper, purple_lower, purple_upper = find_color_ranges()
    
    cap_idx = 0 #camera_indexes[0]
    cap = cv2.VideoCapture(cap_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    canvas_width, canvas_height = int(screen_width), int(screen_height)
    scale_factor = min(canvas_width / screen_width, canvas_height / screen_height)
    
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_window_name = 'Canvas'
    
    last_point_red = None
    last_point_green = None
    last_point_purple = None
    
    cv2.namedWindow(canvas_window_name, cv2.WINDOW_NORMAL)
    monitors = get_monitors()
    if len(monitors) > 1:
        external_monitor = monitors[1]
        cv2.moveWindow(canvas_window_name, external_monitor.x, external_monitor.y)

    
    cv2.setWindowProperty(canvas_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_cnt = 0
    FILL_THRESHOLD_PERCENT = .4
    HOST = 'localhost'
    PORT = 9999
    party_mode_end_time = None

    skewed_clear_area_rect = None
    background_image = None
    skewed_palette_pos = []
    curr_image = None
    clear_area_start_time = None
    clear_area_rect = [(20, 20), (90, 20), 
                       (20, 55), (90, 55)]
    red = (0, 0, 255)
    green = (0, 255, 0)
    purple = (255, 0, 255)

    palette_positions = [[(1250, (40 * y) - 65), (1300, (40 * y) - 65), (1250, (40 * y) - 15), (1300, (40 * y) - 15)] for y in range (1, 14) if y % 2 == 0]
    palette_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    current_color_red = (0, 0, 255)  # Default red laser color
    current_color_green = (0, 255, 0)  # Default green laser color
    current_color_purple = (255, 0, 255)

    #UC03
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind((HOST, PORT))
        server_sock.listen()
        server_sock.setblocking(False)
        while True:
            
            try:
                conn, addr = server_sock.accept()
                client_thread = threading.Thread(target=handle_client_connection, args=(conn, image_queue, command_queue))
                client_thread.start()
            except Exception as e:
                pass
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            if not SKEWED and np.sum(frame) != 0 and np.sum(frame) != 45619200:
                points = select_four_corners(frame.copy())
                get_skew_matrix(points, int(screen_width), int(screen_height))
                clear_area_rect = [(x + points[0][0], y+points[0][1]) for (x, y) in clear_area_rect]
                skewed_clear_area_rect = get_clear_button(matrix, clear_area_rect)
                for palette_pos, palette_color in zip(palette_positions, palette_colors):
                    palette_pos = [(x + points[0][0], y+points[0][1]) for (x, y) in palette_pos]
                    skewed_palette_pos.append(get_color_palette(matrix, palette_pos))

                
            if not command_queue.empty():
                command = command_queue.get()
                if command == 'fill':
                    if not image_queue.empty():
                        clear_canvas(canvas)
                        mode = 'fill'
                        curr_image = image_queue.get()
                elif command == 'party':
                    clear_canvas(canvas)
                    mode = 'party'
                    party_mode_end_time = datetime.now() + timedelta(seconds=30)
                else:
                    mode = 'free' 

            if mode == 'fill':
                if curr_image:
                    background_image = load_scaled_image(curr_image, canvas_width, canvas_height)
                    background_image = background_image[:, :, :3]
                else:
                    JsonResponse({'message': 'Image not loaded successfully.'}, status=405)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            if SKEWED:
                for color_index, color in enumerate(['red', 'green', 'purple']):
                    mask = None
                    if color == 'green':
                        mask = cv2.inRange(hsv_frame, green_lower, green_upper)
                    elif color == 'red':
                        mask = cv2.inRange(hsv_frame, red_lower, red_upper)
                    else:
                        mask = cv2.inRange(hsv_frame, purple_lower, purple_upper)
                        
                    (_, _, _, maxLoc) = cv2.minMaxLoc(mask)
                    (cx, cy) = maxLoc
                    if cx is None or cy is None:
                        continue
                    current_point = skew_point((cx, cy), matrix)
                    if skewed_clear_area_rect[0][0] <= current_point[0] <= skewed_clear_area_rect[1][0] and skewed_clear_area_rect[0][1] <= current_point[1] <= skewed_clear_area_rect[2][1]:
                        if clear_area_start_time is None:
                            clear_area_start_time = time.time()
                        elif time.time() - clear_area_start_time >= 3:
                            clear_canvas(canvas)
                            clear_area_start_time = None

                    if mode == 'fill' and background_image is not None:
                        update_canvas_with_image(canvas, background_image, cx, cy, scale_factor)

                        if frame_cnt % 240 == 0:
                            filled_pixels, total_pixels = count_filled_pixels(canvas, background_image)
                            fill_percentage = filled_pixels / total_pixels
                        else:
                            fill_percentage = 0

                        if fill_percentage >= FILL_THRESHOLD_PERCENT:
                            apply_glitter_effect(canvas, background_image)
                            canvas[:, :] = background_image[:, :]
                            time.sleep(2)
                            background_image = None
                            curr_image = None
                            mode = 'free'
                    elif mode == 'free' or mode == 'party':
                        if mode_status == 'offline':
                            for skewed_palette_p, palette_color in zip(skewed_palette_pos, palette_colors):
                                    top_left, top_right, bot_left, _ = skewed_palette_p
                                    if top_left[0] <= current_point[0] <= top_right[0] and top_left[1] <= current_point[1] <= bot_left[1]:
                                        if color_index == 0:  # Red laser
                                            current_color_red = palette_color
                                        elif color_index == 1:  # Green laser
                                            current_color_green = palette_color
                                        else:
                                            current_color_purple = palette_color
                                        break
                        color = current_color_red if color_index == 0 else current_color_green if color_index == 1 else current_color_purple
                        if color_index == 0:
                            last_point_red = smooth_drawing(last_point_red, current_point, canvas, color)
                        elif color_index == 1:
                            last_point_green = smooth_drawing(last_point_green, current_point, canvas, color)
                        else:
                            last_point_purple = smooth_drawing(last_point_purple, current_point, canvas, color)


            if mode == 'party':
                remaining_time = max(party_mode_end_time - datetime.now(), timedelta(seconds=0)).seconds

                timer_text = f"Time: {remaining_time}"
                text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                timer_position = (canvas_width - text_size[0] - 50, canvas_height - 30)
                cv2.rectangle(canvas, (timer_position[0] - 20, canvas_height - text_size[1] - 60), (canvas_width, canvas_height), (0, 0, 0), -1)
                cv2.putText(canvas, timer_text, timer_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


                if remaining_time == 0:
                    clear_canvas(canvas)
                    mode = 'free'
                    party_mode_end_time = None

            if SKEWED:
                if mode_status == 'offline' and mode == 'free':
                    for skewed_palette_p, palette_color in zip(skewed_palette_pos, palette_colors):
                        draw_palette_box(canvas, skewed_palette_p, palette_color)
                        
                if mode in ['free', 'party']:
                    # Draw the skewed clear button on the canvas
                    if mode == 'party':
                        skewed_canvas = canvas
                    else:
                        skewed_canvas = draw_clear_button(canvas, skewed_clear_area_rect)

                    # Display the canvas with the clear button
                    cv2.imshow(canvas_window_name, skewed_canvas)
                elif mode == 'fill':
                    cv2.imshow(canvas_window_name, canvas)

            frame_cnt += 1
                                            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                absolute_path = os.path.abspath('./../virtual_graffiti/virtual_graffiti/temp/reset_signal.txt')
                try:
                    with open(absolute_path, 'w') as f:
                        f.seek(0)
                        f.write('1')
                    with open("color_ranges.txt", "w") as f:
                        f.seek(0)
                        f.write(f"red_lower: {red_lower}\n")
                        f.write(f"red_upper: {red_upper}\n")
                        f.write(f"green_lower: {green_lower}\n")
                        f.write(f"green_upper: {green_upper}\n")
                        f.write(f"purple_lower: {purple_lower}\n")
                        f.write(f"purple_upper: {purple_upper}\n")
                except Exception as e:
                    print(e)
                break
            elif key == ord('c') and mode != 'party':  # Clear canvas if 'c' is pressed
                clear_canvas(skewed_canvas)
        conn.close()
        cv2.destroyAllWindows()
    cap.release()
        
if __name__ == "__main__":
    init()