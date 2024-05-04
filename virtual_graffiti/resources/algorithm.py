from django.http import JsonResponse
from screeninfo import get_monitors
import cv2
import threading
import numpy as np
from queue import Queue
import os 
import json
import time
from datetime import datetime, timedelta
import socket
import sys
import threading
import requests

def inject_custom_imports():
    """
    Dynamically imports helper functions required for the main script.

    This function imports helper functions specified in the `import_list` and makes them available globally.

    Note:
        This function should be called before accessing any helper functions.

    """
    import_list = [
        "helpers.calibration.find_color_ranges",
        "helpers.calibration.select_four_corners",
        "helpers.calibration.get_skew_matrix",
        "helpers.canvas_and_modes.load_scaled_image",
        "helpers.canvas_and_modes.update_canvas_with_image",
        "helpers.canvas_and_modes.apply_glitter_effect",
        "helpers.canvas_and_modes.get_clear_button",
        "helpers.canvas_and_modes.get_color_palette",
        "helpers.canvas_and_modes.draw_clear_button",
        "helpers.canvas_and_modes.draw_palette_box",
        "helpers.canvas_and_modes.clear_canvas",
        "helpers.computing.count_filled_pixels",
        "helpers.computing.smooth_drawing",
        "helpers.computing.skew_point"
        ""
    ]
    for import_str in import_list:
        module_name, obj_name = import_str.rsplit(".", 1)
        module = __import__(module_name, fromlist=[obj_name])
        globals()[obj_name] = getattr(module, obj_name)

def handle_client_connection(conn, image_queue, command_queue):
    """
    Handles communication with a client.

    This function listens for incoming data from a client connection and processes commands accordingly.

    Parameters:
        conn (socket.socket): The client connection socket.
        image_queue (Queue): A queue for receiving image data from the client.
        command_queue (Queue): A queue for receiving commands from the client.

    """
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

def hex_to_bgr(hex_color):
    """
    Converts a hexadecimal color code to BGR format.

    Parameters:
        hex_color (str): A hexadecimal color code (e.g., "#RRGGBB").

    Returns:
        tuple: A tuple containing the BGR color values.
    """
    rgb_color = np.array([int(hex_color[i:i+2], 16) for i in (1, 3, 5)])
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    return bgr_color

#FR1, UC10
def init(mode_status='offline'):
    """
    Initializes the main script.

    This function sets up communication with clients, initializes camera and display settings,
    and manages the main processing loop.

    Parameters:
        mode_status (str, optional): The status of the mode (offline/online). Defaults to 'offline'.
    """
    inject_custom_imports() # due to popen subprocess opening new area.
    skewed = False
    matrix = None
    mode = 'free'
    image_queue = Queue()
    command_queue = Queue()
    
    """
    camera_indexes = enumerate_cameras()
    print(camera_indexes)
    if len(camera_indexes) == 0:
        print('No cameras found')
        return
    print('Camera found')
    """

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
    
    cv2.namedWindow(canvas_window_name, cv2.WND_PROP_FULLSCREEN + cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(canvas_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    monitors = get_monitors()
    if len(monitors) > 1:
        external_monitor = monitors[1]
        cv2.moveWindow(canvas_window_name, external_monitor.x, external_monitor.y)    

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

    palette_positions = [[(0, (50 * y)), (50, (50 * y)), (0, (50 * y) + 56), (50, (50 * y) + 56)] for y in range (1, 14) if y % 2 == 0]
    palette_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    current_color_red = (0, 0, 255)
    current_size_red = 2
    current_color_green = (0, 255, 0)
    current_size_green = 2
    current_color_purple = (255, 0, 255)
    current_size_purple = 2

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

            if not skewed and np.sum(frame) != 0 and np.sum(frame) != 45619200:
                points = select_four_corners(frame.copy())
                matrix = get_skew_matrix(points, int(screen_width), int(screen_height))
                skewed = True
                clear_area_rect = [(x + points[0][0], y+points[0][1]) for (x, y) in clear_area_rect]
                skewed_clear_area_rect = get_clear_button(matrix, clear_area_rect)
                for palette_pos, palette_color in zip(palette_positions, palette_colors):
                    palette_pos = [(x + points[1][0] - ((points[1][0] - points[1][1]) * .1), y+points[1][1]) for (x, y) in palette_pos]
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

            if mode_status == 'online' and frame_cnt % 300 == 0:
                response = requests.get('http://localhost:8001/get_lasers/')
                if response.status_code == 200:
                    laser_data = response.json()
                    if 'Green' in laser_data:
                        current_color_green = hex_to_bgr(laser_data['Green'][0])
                        current_size_green = laser_data['Green'][1]
                        print(f'COLOR: {current_color_green}')
                else:
                    print("Failed to fetch laser data.")
    
            if skewed:
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
                                        if color_index == 0:
                                            current_color_red = palette_color
                                        elif color_index == 1:
                                            current_color_green = palette_color
                                        else:
                                            current_color_purple = palette_color
                                        break
                        color = current_color_red if color_index == 0 else current_color_green if color_index == 1 else current_color_purple
                        print(f'{color_index}, color: {color}')
                        if color_index == 0:
                            last_point_red = smooth_drawing(last_point_red, current_point, canvas, color, thickness=current_size_red)
                        elif color_index == 1:
                            last_point_green = smooth_drawing(last_point_green, current_point, canvas, color, thickness=current_size_green)
                        else:
                            last_point_purple = smooth_drawing(last_point_purple, current_point, canvas, color, thickness=current_size_purple)


            if mode == 'party':
                remaining_time = max(party_mode_end_time - datetime.now(), timedelta(seconds=0)).seconds

                timer_text = f"Time: {remaining_time}"
                text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                timer_position = (canvas_width - text_size[0] - 50, canvas_height)
                cv2.rectangle(canvas, (timer_position[0] - 20, canvas_height - text_size[1] - 60), (canvas_width, canvas_height), (0, 0, 0), -1)
                cv2.putText(canvas, timer_text, timer_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


                if remaining_time == 0:
                    clear_canvas(canvas)
                    mode = 'free'
                    party_mode_end_time = None

            if skewed:
                if mode_status == 'offline' and (mode == 'free' or  mode == 'party'):
                    for skewed_palette_p, palette_color in zip(skewed_palette_pos, palette_colors):
                        draw_palette_box(canvas, skewed_palette_p, palette_color)
                        
                if mode in ['free', 'party']:
                    if mode == 'party':
                        skewed_canvas = canvas
                    else:
                        skewed_canvas = draw_clear_button(canvas, skewed_clear_area_rect)
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
                    color_ranges = {
                        "red": {
                            "lower": red_lower.tolist(),
                            "upper": red_upper.tolist()
                        },
                        "green": {
                            "lower": green_lower.tolist(),
                            "upper": green_upper.tolist()
                        },
                        "purple": {
                            "lower": purple_lower.tolist(),
                            "upper": purple_upper.tolist()
                        }
                    }
                    with open("color_ranges.txt", "w") as f:
                        json.dump(color_ranges, f)

                except Exception as e:
                    print(e)
                server_sock.close()
                break
            elif key == ord('c') and mode != 'party':
                clear_canvas(skewed_canvas)
        cv2.destroyAllWindows()
    cap.release()
        
if __name__ == "__main__":
    mode_status = sys.argv[1]
    init(mode_status)