import ctypes
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import numpy as np
import math
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from pynput import mouse
import cv2
import logging
import mss
import configparser
import random

# initialize configparser
config = configparser.ConfigParser()

# global variables
aimbot_enabled = False
aimbot_on_key_enabled = False
aimbot_key = None
model = None
offset_x = 0
offset_y = 0
aimbot_strength = 1.5
update_interval = 0.03
interpolation_steps = 50
fov_percentage = 50
key_pressed = False
use_gpu = torch.cuda.is_available()
fov_overlay_enabled = True
fov_drawn = False
console_silent = False
frame_capture_percentage = 100
target_fps = 60

# overlay window for drawing
overlay_window = None
canvas = None

# constants for mouse events
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000

# logging for yolo
yolo_logger = logging.getLogger("ultralytics")
yolo_logger.setLevel(logging.WARNING)

# save configuration
def save_config():
    file_path = filedialog.asksaveasfilename(defaultextension=".cfg", filetypes=[("Config Files", "*.cfg")])
    if not file_path:
        return

    config['Settings'] = {
        'offset_x': offset_x_slider.get(),
        'offset_y': offset_y_slider.get(),
        'fov_percentage': fov_slider.get(),
        'aimbot_strength': strength_slider.get(),
        'update_interval': update_interval_slider.get(),
        'interpolation_steps': interpolation_steps_slider.get(),
        'frame_capture_percentage': frame_capture_slider.get()
    }

    with open(file_path, 'w') as configfile:
        config.write(configfile)

    messagebox.showinfo("success", "configuration saved")

# load configuration
def load_config():
    file_path = filedialog.askopenfilename(filetypes=[("Config Files", "*.cfg")])
    if not file_path:
        return

    config.read(file_path)

    try:
        offset_x_slider.set(int(config['Settings']['offset_x']))
        offset_y_slider.set(int(config['Settings']['offset_y']))
        fov_slider.set(int(config['Settings']['fov_percentage']))
        strength_slider.set(float(config['Settings']['aimbot_strength']))
        update_interval_slider.set(float(config['Settings']['update_interval']))
        interpolation_steps_slider.set(int(config['Settings']['interpolation_steps']))
        frame_capture_slider.set(int(config['Settings']['frame_capture_percentage']))

        messagebox.showinfo("success", "configuration loaded")
    except KeyError as e:
        messagebox.showerror("error", f"invalid config file: missing {str(e)}")

# print log to console
def print_log(message, max_length=200):
    global console_silent
    if console_silent:
        return

    if len(message) > max_length:
        message = message[:max_length] + '...'

    print(message)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

# send mouse movement
def send_mouse_event(dx, dy):
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, None)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# get cursor position
def get_cursor_pos():
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

# create the overlay window
def create_overlay():
    global overlay_window, canvas, screen_width, screen_height

    if overlay_window is None:
        overlay_window = tk.Toplevel()
        overlay_window.overrideredirect(True)
        overlay_window.attributes("-topmost", True)
        overlay_window.attributes("-transparentcolor", "white")

        overlay_window.geometry(f"{screen_width}x{screen_height}+0+0")
        canvas = tk.Canvas(overlay_window, bg='white', highlightthickness=0, width=screen_width, height=screen_height)
        canvas.pack(fill="both", expand=True)
        canvas.config(width=screen_width, height=screen_height)

# clear overlay canvas
def clear_overlay():
    if canvas is not None:
        canvas.delete("all")

screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

# check if within fov
def is_within_fov(x, y):
    center_x = screen_width // 2
    center_y = screen_height // 2
    distance = math.hypot(x - center_x, y - center_y)
    fov_radius = (screen_height * fov_percentage) / 100
    return distance <= fov_radius

# move the mouse with smooth motion
def human_like_mouse_movement(start_x, start_y, target_x, target_y, strength):
    distance = math.hypot(target_x - start_x, target_y - start_y)
    steps = max(10, int(distance / 10))
    dx = (target_x - start_x) / steps
    dy = (target_y - start_y) / steps

    for step in range(steps):
        jitter_x = random.uniform(-1, 1) * strength * 0.05
        jitter_y = random.uniform(-1, 1) * strength * 0.05

        send_mouse_event(int(dx + jitter_x), int(dy + jitter_y))
        time.sleep(update_interval)

# move mouse to target
def move_mouse_to_target(bbox):
    global screen_width, screen_height

    x1, y1, x2, y2 = bbox
    target_center_x = int((x1 + x2) / 2)
    target_center_y = int((y1 + y2) / 2)

    target_center_x += offset_x
    target_center_y += offset_y

    actual_screen_width, actual_screen_height = get_screen_resolution()

    scale_x = actual_screen_width / resize_width
    scale_y = actual_screen_height / resize_height

    target_center_x = int(target_center_x * scale_x)
    target_center_y = int(target_center_y * scale_y)

    if is_within_fov(target_center_x, target_center_y):
        start_x, start_y = get_cursor_pos()
        human_like_mouse_movement(start_x, start_y, target_center_x, target_center_y, aimbot_strength)

# draw fov overlay
def draw_fov():
    global screen_width, screen_height, fov_percentage, fov_drawn
    create_overlay()

    if not fov_drawn:
        actual_screen_width, actual_screen_height = get_screen_resolution()
        center_x = actual_screen_width // 2
        center_y = actual_screen_height // 2
        fov_radius = (actual_screen_height * fov_percentage) / 100

        canvas.create_oval(center_x - fov_radius, center_y - fov_radius,
                           center_x + fov_radius, center_y + fov_radius,
                           outline="green", width=2)
        fov_drawn = True

# load the yolo model
def load_model(path):
    global model, use_gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(path).to(device)
    use_gpu = torch.cuda.is_available()
    print_log(f"model {path} loaded on {device}")

resize_width = 640
resize_height = 360

# capture screen region
def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        return resized_frame

# detect objects using yolo
def detect_objects(image):
    global use_gpu
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    if use_gpu:
        image_tensor = image_tensor.cuda()

    height, width = image_tensor.shape[2], image_tensor.shape[3]
    new_height = height - (height % 32)
    new_width = width - (width % 32)

    image_tensor = F.interpolate(image_tensor, size=(new_height, new_width)).cuda()

    results = model(image_tensor)
    detections = results[0]

    detected_boxes = []
    for detection in detections.boxes:
        bbox = detection.xyxy.cpu().numpy()[0]
        class_id = int(detection.cls.cpu().numpy()[0])
        label = model.names[class_id]

        if class_id == 0 and label == "player":
            detected_boxes.append(bbox)

    return detected_boxes

# get screen resolution
def get_screen_resolution():
    hdc = ctypes.windll.user32.GetDC(0)
    width = ctypes.windll.gdi32.GetDeviceCaps(hdc, 118)
    height = ctypes.windll.gdi32.GetDeviceCaps(hdc, 117)
    return width, height

# find closest target
def prioritize_targets(detections):
    return min(detections, key=lambda bbox: math.hypot(
        (bbox[0] + bbox[2]) / 2 - screen_width // 2,
        (bbox[1] + bbox[3]) / 2 - screen_height // 2))

# adjust aimbot strength based on distance
def adjust_aimbot_strength_based_on_distance(distance):
    if distance < 100:
        return 0.8
    elif distance < 300:
        return 1.2
    else:
        return 1.5
def draw_target_square(bbox):
    if canvas is None:
        return

    x1, y1, x2, y2 = bbox
    actual_screen_width, actual_screen_height = get_screen_resolution()

    # scale to match screen res
    scale_x = actual_screen_width / resize_width
    scale_y = actual_screen_height / resize_height

    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # draw green square
    canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)

# run aimbot loop
def run_aimbot():
    global aimbot_enabled, key_pressed, fov_overlay_enabled, frame_capture_percentage
    frame_counter = 0
    target_frame_time = 1 / target_fps

    while aimbot_enabled:
        frame_counter += 1
        start_time = time.time()

        if frame_counter % (100 // frame_capture_percentage) != 0:
            time.sleep(update_interval)
            continue

        screen = capture_screen()
        detections = detect_objects(screen)

        detected_players = []

        if detections:
            detected_players = detections

        if fov_overlay_enabled and not fov_drawn:
            draw_fov()

        # clear other overlays
        clear_overlay()

        if detected_players and (not aimbot_on_key_enabled or (aimbot_on_key_enabled and key_pressed)):
            closest_player = prioritize_targets(detected_players)
            move_mouse_to_target(closest_player)

            # draw for closest player
            draw_target_square(closest_player)

        elapsed_time = time.time() - start_time
        sleep_time = target_frame_time - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

            

# threaded aimbot
def threaded_aimbot():
    capture_thread = threading.Thread(target=run_aimbot, daemon=True)
    capture_thread.start()

# toggle fov overlay
def toggle_fov_overlay():
    global fov_overlay_enabled, fov_drawn
    fov_overlay_enabled = not fov_overlay_enabled
    if fov_overlay_enabled:
        toggle_fov_button.config(text="disable fov overlay")
        fov_drawn = False
    else:
        toggle_fov_button.config(text="enable fov overlay")
        clear_overlay()

# toggle aimbot on/off
def toggle_aimbot():
    global aimbot_enabled
    if aimbot_enabled:
        aimbot_enabled = False
        toggle_button.config(text="enable aimbot")
        clear_overlay()
    else:
        aimbot_enabled = True
        threaded_aimbot()

# adjust x offset
def adjust_offset_x(val):
    global offset_x
    offset_x = int(val)
    print_log(f"offset x: {offset_x}")

# adjust y offset
def adjust_offset_y(val):
    global offset_y
    offset_y = int(val)
    print_log(f"offset y: {offset_y}")

# adjust fov percentage
def adjust_fov(val):
    global fov_percentage
    fov_percentage = int(val)
    print_log(f"fov: {fov_percentage}%")

# adjust aimbot strength
def adjust_aimbot_strength(val):
    global aimbot_strength
    aimbot_strength = float(val)
    print_log(f"aimbot strength: {aimbot_strength}")

# adjust update interval
def adjust_update_interval(val):
    global update_interval
    update_interval = float(val)
    print_log(f"update interval: {update_interval}")

# adjust interpolation steps
def adjust_interpolation_steps(val):
    global interpolation_steps
    interpolation_steps = int(val)
    print_log(f"interpolation steps: {interpolation_steps}")

# adjust frame capture percentage
def adjust_frame_capture_percentage(val):
    global frame_capture_percentage
    frame_capture_percentage = int(val)
    print_log(f"frame capture: {frame_capture_percentage}%")

# select yolo model
def select_model():
    global model_path
    model_path = filedialog.askopenfilename(filetypes=[("YOLOv8 Model", "*.pt")])
    if model_path:
        load_model(model_path)
        model_label.config(text=f"model: {model_path.split('/')[-1]}")

# toggle console logging
def toggle_console_logging():
    global console_silent, yolo_logger
    console_silent = not console_silent
    toggle_console_button.config(text="enable console logging" if console_silent else "disable console logging")
    
    if console_silent:
        yolo_logger.setLevel(logging.WARNING)
    else:
        yolo_logger.setLevel(logging.INFO)

# mouse click detection
def on_click(x, y, button, pressed):
    global key_pressed
    if button == mouse.Button.left and pressed and aimbot_key == "Mouse1":
        key_pressed = True
    elif button == mouse.Button.right and pressed and aimbot_key == "Mouse2":
        key_pressed = True
    elif not pressed:
        key_pressed = False

listener = mouse.Listener(on_click=on_click)
listener.start()

# set aim key
def set_aim_key(event):
    global aimbot_key
    aimbot_key = aim_key_var.get()
    print_log(f"aim key: {aimbot_key}")

# enable or disable aimbot on key press
def enable_aimbot_on_key():
    global aimbot_on_key_enabled
    if aimbot_on_key_enabled:
        aimbot_on_key_enabled = False
        aim_key_dropdown.pack_forget()
    else:
        aimbot_on_key_enabled = True
        aim_key_dropdown.pack(pady=10)

# scrollable frame class for GUI
class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# GUI setup
root = tk.Tk()
root.title("aimbot control panel")
root.geometry("400x600")

scrollable_frame = ScrollableFrame(root)
scrollable_frame.pack(fill="both", expand=True)

# add widgets
toggle_button = tk.Button(scrollable_frame.scrollable_frame, text="enable aimbot", command=toggle_aimbot, width=25)
toggle_button.pack(pady=10)

model_button = tk.Button(scrollable_frame.scrollable_frame, text="select yolov8 model", command=select_model, width=25)
model_button.pack(pady=10)

model_label = tk.Label(scrollable_frame.scrollable_frame, text="no model selected")
model_label.pack(pady=5)

offset_x_slider = tk.Scale(scrollable_frame.scrollable_frame, from_=-200, to=200, orient="horizontal", label="offset x", command=adjust_offset_x)
offset_x_slider.pack(pady=10)

offset_y_slider = tk.Scale(scrollable_frame.scrollable_frame, from_=-200, to=200, orient="horizontal", label="offset y", command=adjust_offset_y)
offset_y_slider.pack(pady=10)

fov_slider = tk.Scale(scrollable_frame.scrollable_frame, from_=0, to=100, orient="horizontal", label="fov percentage", command=adjust_fov)
fov_slider.set(50)
fov_slider.pack(pady=10)

strength_slider = tk.Scale(scrollable_frame.scrollable_frame, from_=0.1, to=10, orient="horizontal", resolution=0.01, label="aimbot strength", command=adjust_aimbot_strength)
strength_slider.set(1.5)
strength_slider.pack(pady=10)

update_interval_slider = tk.Scale(scrollable_frame.scrollable_frame, from_=0.001, to=0.1, resolution=0.001, orient="horizontal", label="update interval", command=adjust_update_interval)
update_interval_slider.set(0.01)
update_interval_slider.pack(pady=10)

interpolation_steps_slider = tk.Scale(scrollable_frame.scrollable_frame, from_=10, to=500, orient="horizontal", label="interpolation steps", command=adjust_interpolation_steps)
interpolation_steps_slider.set(50)
interpolation_steps_slider.pack(pady=10)

frame_capture_slider = tk.Scale(scrollable_frame.scrollable_frame, from_=1, to=100, orient="horizontal", label="frame capture percentage", command=adjust_frame_capture_percentage)
frame_capture_slider.set(100)
frame_capture_slider.pack(pady=10)

aimbot_on_key_checkbox = tk.Checkbutton(scrollable_frame.scrollable_frame, text="aimbot on key", command=enable_aimbot_on_key)
aimbot_on_key_checkbox.pack(pady=10)

aim_key_var = tk.StringVar(value="Mouse1")
aim_key_dropdown = ttk.Combobox(scrollable_frame.scrollable_frame, textvariable=aim_key_var, values=["Mouse1", "Mouse2"])
aim_key_dropdown.bind("<<ComboboxSelected>>", set_aim_key)
aim_key_dropdown.pack(pady=10)

toggle_fov_button = tk.Button(scrollable_frame.scrollable_frame, text="disable fov overlay", command=toggle_fov_overlay, width=25)
toggle_fov_button.pack(pady=10)

toggle_console_button = tk.Button(scrollable_frame.scrollable_frame, text="disable console logging", command=toggle_console_logging, width=25)
toggle_console_button.pack(pady=10)

save_button = tk.Button(scrollable_frame.scrollable_frame, text="save config", command=save_config, width=25)
save_button.pack(pady=10)

load_button = tk.Button(scrollable_frame.scrollable_frame, text="load config", command=load_config, width=25)
load_button.pack(pady=10)

root.mainloop()
