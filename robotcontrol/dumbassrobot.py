import tkinter as tk
import RPi.GPIO as GPIO
import board
import busio
import time
import threading
import cv2
import random
import queue
from PIL import Image, ImageTk
from ultralytics import YOLO
from adafruit_pca9685 import PCA9685

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except:
            pass
        self.cap.release()

stream = None

IN1 = 17
IN2 = 27
IN3 = 22
IN4 = 23

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)

i2c = busio.I2C(board.SCL, board.SDA)
try:
    pca = PCA9685(i2c)
    pca.frequency = 50
    hardware_available = True
    print("PCA9685 initialized successfully.")
except Exception as e:
    print(f"Hardware not found or library missing. Running in simulation mode. Error: {e}")
    hardware_available = False

ENA_CHANNEL = 14
ENB_CHANNEL = 15

try:
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16, i2c=i2c) if hardware_available else None
except Exception as e:
    print(f"ServoKit not found. Error: {e}")
    kit = None

current_angles = {i: 90.0 for i in range(4)}
target_angles = {0: 60.0, 1: 0.0, 2: 90.0, 3: 90.0}
global_root = None

RUN_TIME_MS = 500
MOTOR_SPEED = 32

KP = 0.07
KI = 0.002
KD = 0.02
BASE_SPEED = 44
SEARCH_SPEED = 32
STEER_RIGHT_BIAS = 0.7
MOTOR_PULSE_S = 0.15
MOTOR_WAIT_S = 0.3
ALIGN_TOLERANCE_PX = 45
ALIGN_TOUCH_WAIT_S = 0.8
BACK_AWAY_VERIFY_S = 1.0
BACK_AWAY_COOLDOWN_S = 1.0
GRAB_MIN_DIST = 15.0
GRAB_MAX_DIST = 22.0

btn_auto = None
video_label = None
frame_queue = queue.Queue(maxsize=1)

last_stop_time = 0.0

def enforce_stop_delay():
    elapsed = time.time() - last_stop_time
    if elapsed < 0.4:
        time.sleep(0.4 - elapsed)

def set_speed(speed_percent, bypass_delay=False):
    if not hardware_available:
        return
    if speed_percent > 0 and not bypass_delay:
        enforce_stop_delay()
    duty_cycle = int((speed_percent / 100.0) * 0xffff)
    pca.channels[ENA_CHANNEL].duty_cycle = duty_cycle
    pca.channels[ENB_CHANNEL].duty_cycle = duty_cycle

def set_motor_speeds(left_speed_percent, right_speed_percent, bypass_delay=False):
    if not hardware_available:
        return
    if (left_speed_percent > 0 or right_speed_percent > 0) and not bypass_delay:
        enforce_stop_delay()
    left_speed_percent = max(0, min(100, left_speed_percent))
    right_speed_percent = max(0, min(100, right_speed_percent))

    duty_cycle_left = int((left_speed_percent / 100.0) * 0xffff)
    duty_cycle_right = int((right_speed_percent / 100.0) * 0xffff)
    pca.channels[ENA_CHANNEL].duty_cycle = duty_cycle_left
    pca.channels[ENB_CHANNEL].duty_cycle = duty_cycle_right

def forward():
    print("Moving Forward")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(MOTOR_SPEED)

def backward():
    print("Moving Backward")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    set_speed(MOTOR_SPEED)

def left():
    print("Turning Left")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(MOTOR_SPEED)

def right():
    print("Turning Right")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    set_speed(MOTOR_SPEED)

def stop(set_delay=True):
    print("Stopping")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

    if hardware_available:
        pca.channels[ENA_CHANNEL].duty_cycle = 0
        pca.channels[ENB_CHANNEL].duty_cycle = 0

    if set_delay:
        global last_stop_time
        last_stop_time = time.time()

def stop_button_pressed():
    global auto_mode
    auto_mode = False
    if btn_auto is not None:
        try:
            btn_auto.config(text="Auto Mode: OFF", bg="SystemButtonFace")
        except Exception:
            pass
    stop()

def forward_timed():
    forward()
    global_root.after(RUN_TIME_MS, stop)

def backward_timed():
    backward()
    global_root.after(RUN_TIME_MS, stop)

def left_timed():
    left()
    global_root.after(RUN_TIME_MS, stop)

def right_timed():
    right()
    global_root.after(RUN_TIME_MS, stop)

def update_servo_task(root):
    step = 1.0
    delay_ms = 20

    for i in range(4):
        target = target_angles[i]
        current = current_angles[i]

        if current != target:
            diff = target - current
            if abs(diff) <= step:
                current_angles[i] = target
            else:
                current_angles[i] += step if diff > 0 else -step

            if hardware_available and kit is not None:
                try:
                    kit.servo[i].angle = current_angles[i]
                except Exception as e:
                    pass

    root.after(delay_ms, update_servo_task, root)

def start():
    def step0():
        target_angles[0] = 60.0
        check_and_next(0, 60.0, step1)
    def step1():
        target_angles[1] = 0.0
        check_and_next(1, 0.0, step2)
    def step2():
        target_angles[2] = 130.0
        check_and_next(2, 130.0, step3)
    def step3():
        target_angles[3] = 90.0
    step0()

def check_and_next(servo_index, target, next_step_func):
    if current_angles[servo_index] == target:
        next_step_func()
    else:
        global_root.after(50, check_and_next, servo_index, target, next_step_func)

def grab_pos():
    target_angles[0] = 90.0
    target_angles[1] = 90.0
    target_angles[2] = 90.0
    target_angles[3] = 90.0

def alt_grab_pos():
    target_angles[0] = 75.0
    target_angles[1] = 90.0
    target_angles[2] = 90.0
    target_angles[3] = 90.0

def grab():
    target_angles[3] = 180.0

auto_mode = False
def toggle_auto(btn):
    global auto_mode
    auto_mode = not auto_mode
    btn.config(text=f"Auto Mode: {'ON' if auto_mode else 'OFF'}", bg="green" if auto_mode else "SystemButtonFace")
    print(f"Auto Mode: {'ON' if auto_mode else 'OFF'}")

def auto_loop():
    global stream
    model = YOLO("/home/milm/work/6th.pt")
    stream = CameraStream(0)
    dimensions = {
        'Plastic': [23.5, 6.5, 6.5],
        'Box': [10.0, 7.0, 3.3],
        'Cardboard': [12.4, 8.3, 5.0],
        'Bottle': [20.0, 6.5, 6.5]
    }
    focal_length = 1234.09

    pid_error_sum = 0.0
    pid_last_error = 0.0
    pid_last_time = time.time()

    search_direction = None

    last_target_seen_time = 0.0
    last_known_target = None

    is_verifying = False
    verification_start_time = 0.0

    too_close_start_time = 0.0
    last_back_away_time = 0.0
    centering_turns = 0

    def trigger_grab_sequence(cls):
        nonlocal last_known_target, last_target_seen_time, pid_error_sum, pid_last_error, pid_last_time, is_verifying, centering_turns
        stop()

        last_known_target = None
        last_target_seen_time = 0.0

        pid_error_sum = 0.0
        pid_last_error = 0.0
        pid_last_time = time.time()
        is_verifying = False
        centering_turns = 0

        print(f"Target aligned and verified. Initiating grab sequence for {cls}.")

        start()
        sleep_and_update_camera(2.5, f"Arm to Start ({cls})")

        if cls == "Plastic":
            alt_grab_pos()
        else:
            grab_pos()
        sleep_and_update_camera(2.0, "Moving to Grab Position")

        grab()
        sleep_and_update_camera(1.5, "Grabbing Object")

        is_sorted = (cls not in ["Bottle", "Plastic"])
        if is_sorted:
            left()
            sleep_and_update_camera(1.0, "Sorting: turning left")
            stop()

        start()
        sleep_and_update_camera(2.5, "Returning to Start")

        if is_sorted:
            right()
            sleep_and_update_camera(1.0, "Returning to search heading")
            stop()

        stop()

        sleep_and_update_camera(10.0, "Post-Grab Pause (10s)")

    def sleep_and_update_camera(duration, msg):
        start_t = time.time()
        while time.time() - start_t < duration:
            succ, f_sleep = stream.read()
            if succ and f_sleep is not None:
                cv2.putText(f_sleep, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                try:
                    small_f = cv2.resize(f_sleep, (320, 240))
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    frame_queue.put_nowait(small_f)
                except Exception:
                    pass
            time.sleep(0.03)

    while True:
        success, frame = stream.read()
        if not success or frame is None:
            time.sleep(0.01)
            continue

        if not auto_mode:
            try:
                small_frame = cv2.resize(frame, (320, 240))
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(small_frame)
            except Exception:
                pass

            pid_error_sum = 0.0
            pid_last_error = 0.0
            pid_last_time = time.time()
            search_direction = None
            last_target_seen_time = time.time()
            last_known_target = None
            is_verifying = False
            verification_start_time = 0.0
            too_close_start_time = 0.0
            centering_turns = 0
            time.sleep(0.03)
            continue

        results = model(frame, stream=True, imgsz=320)
        h, w, _ = frame.shape
        center_x_frame = w / 2
        target = None

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                x1, y1 = float(box.xyxy[0][0]), float(box.xyxy[0][1])
                x2, y2 = float(box.xyxy[0][2]), float(box.xyxy[0][3])
                pw, ph = x2 - x1, y2 - y1

                if pw > 0 and ph > 0:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if class_name in dimensions:
                        dims = dimensions[class_name]

                        if class_name == 'Plastic' or class_name == 'Bottle':
                            if pw > ph:
                                dist = (dims[1] * focal_length) / ph
                            else:
                                dist = (dims[1] * focal_length) / pw
                        else:
                            observed_ratio = pw / ph
                            best_diff = float('inf')
                            best_w, best_h = dims[0], dims[1]
                            possible_pairs = [
                                (dims[0], dims[1]), (dims[0], dims[2]),
                                (dims[1], dims[0]), (dims[1], dims[2]),
                                (dims[2], dims[0]), (dims[2], dims[1])
                            ]
                            for W, H in possible_pairs:
                                expected_ratio = W / H
                                diff = abs(observed_ratio - expected_ratio)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_w, best_h = W, H

                            if best_w >= best_h:
                                dist = (best_w * focal_length) / pw
                            else:
                                dist = (best_h * focal_length) / ph

                        cx = (x1 + x2) / 2

                        cv2.putText(frame, f"{dist:.1f}cm", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        if not target:
                            target = {"class": class_name, "distance": dist, "cx": cx, "x1": x1, "x2": x2, "y1": y1, "y2": y2}

        if target:
            last_target_seen_time = time.time()
            last_known_target = target
            is_using_memory = False
        else:
            if last_known_target is not None and (time.time() - last_target_seen_time) < 1.2:
                target = last_known_target
                is_using_memory = True
            else:
                last_known_target = None
                is_using_memory = False

        if not target:
            if is_verifying:
                print("Verification aborted: target lost.")
                is_verifying = False

            pid_error_sum = 0.0
            pid_last_error = 0.0
            pid_last_time = time.time()
            too_close_start_time = 0.0
            centering_turns = 0

            if search_direction is None:
                stop()
                search_direction = random.choice(["left", "right"])
                print(f"No object detected. Searching: spinning {search_direction}")

            cv2.putText(frame, f"Searching: {search_direction.upper()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            h_f, w_f, _ = frame.shape
            center_x_f = w_f / 2
            cv2.line(frame, (int(center_x_f), 0), (int(center_x_f), h_f), (0, 0, 255), 2)

            try:
                small_frame = cv2.resize(frame, (320, 240))
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(small_frame)
            except Exception:
                pass

            if search_direction == "left":
                GPIO.output(IN1, GPIO.HIGH)
                GPIO.output(IN2, GPIO.LOW)
                GPIO.output(IN3, GPIO.HIGH)
                GPIO.output(IN4, GPIO.LOW)
                set_speed(SEARCH_SPEED)
            else:
                GPIO.output(IN1, GPIO.LOW)
                GPIO.output(IN2, GPIO.HIGH)
                GPIO.output(IN3, GPIO.LOW)
                GPIO.output(IN4, GPIO.HIGH)
                set_speed(SEARCH_SPEED)
            continue

        search_direction = None

        cx = target["cx"]
        dist = target["distance"]
        cls = target["class"]

        is_aligned = abs(cx - center_x_frame) <= ALIGN_TOLERANCE_PX

        if is_using_memory:
            cv2.rectangle(frame, (int(target["x1"]), int(target["y1"])), (int(target["x2"]), int(target["y2"])), (0, 165, 255), 2)
            cv2.putText(frame, f"{cls} (Memory)", (int(target["x1"]), int(target["y1"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.putText(frame, f"{dist:.1f}cm", (int(target["x1"]), int(target["y2"]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if is_using_memory:
            cv2.putText(frame, f"Tracking (Memory): {cls}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame, f"Tracking: {cls}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        line_color = (0, 255, 0) if is_aligned else (0, 0, 255)
        cv2.line(frame, (int(center_x_frame), 0), (int(center_x_frame), h), line_color, 2)

        try:
            small_frame = cv2.resize(frame, (320, 240))
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(small_frame)
        except Exception:
            pass

        if GRAB_MIN_DIST <= dist <= GRAB_MAX_DIST and is_aligned:
            too_close_start_time = 0.0
            if not is_verifying:
                stop()
                is_verifying = True
                verification_start_time = time.time()
                print("In range and aligned. Verifying alignment for 0.5s...")
            else:
                if time.time() - verification_start_time >= 0.5:
                    trigger_grab_sequence(cls)
                else:
                    stop()
        else:
            if is_verifying:
                print("Verification aborted: target moved or lost alignment.")
                is_verifying = False

            if dist > GRAB_MAX_DIST:
                too_close_start_time = 0.0
                centering_turns = 0
                GPIO.output(IN1, GPIO.LOW)
                GPIO.output(IN2, GPIO.HIGH)
                GPIO.output(IN3, GPIO.HIGH)
                GPIO.output(IN4, GPIO.LOW)

                current_time = time.time()
                dt = current_time - pid_last_time
                if dt <= 0:
                    dt = 0.01

                error = cx - center_x_frame
                pid_error_sum += error * dt
                pid_error_sum = max(-100, min(100, pid_error_sum))

                derivative = (error - pid_last_error) / dt
                pid_output = (KP * error) + (KI * pid_error_sum) + (KD * derivative)

                pid_last_error = error
                pid_last_time = current_time

                if pid_output > 0:
                    pid_output_steer = pid_output * STEER_RIGHT_BIAS
                else:
                    pid_output_steer = pid_output

                left_speed = BASE_SPEED + pid_output_steer
                right_speed = BASE_SPEED - pid_output_steer

                left_speed = max(0, min(BASE_SPEED, left_speed))
                right_speed = max(0, min(BASE_SPEED, right_speed))

                set_motor_speeds(left_speed, right_speed, bypass_delay=True)
                time.sleep(MOTOR_PULSE_S)
                stop(set_delay=False)
                time.sleep(MOTOR_WAIT_S)
            elif dist < GRAB_MIN_DIST:
                pid_error_sum = 0.0
                pid_last_error = 0.0
                pid_last_time = time.time()
                centering_turns = 0

                if too_close_start_time == 0.0:
                    too_close_start_time = time.time()
                    stop(set_delay=False)
                    print(f"Object too close ({dist:.1f}cm). Pausing to verify for {BACK_AWAY_VERIFY_S}s before backing away...")
                else:
                    if time.time() - too_close_start_time >= BACK_AWAY_VERIFY_S:
                        time_since_last_back = time.time() - last_back_away_time
                        if time_since_last_back >= BACK_AWAY_COOLDOWN_S:
                            print(f"Too close ({dist:.1f}cm) and verified. Backing away.")
                            GPIO.output(IN1, GPIO.HIGH)
                            GPIO.output(IN2, GPIO.LOW)
                            GPIO.output(IN3, GPIO.LOW)
                            GPIO.output(IN4, GPIO.HIGH)
                            set_speed(BASE_SPEED, bypass_delay=True)
                            time.sleep(MOTOR_PULSE_S)
                            stop(set_delay=False)
                            time.sleep(MOTOR_WAIT_S)
                            last_back_away_time = time.time()
                        else:
                            stop(set_delay=False)
                    else:
                        stop(set_delay=False)
            elif not is_aligned:
                too_close_start_time = 0.0
                pid_error_sum = 0.0
                pid_last_error = 0.0
                pid_last_time = time.time()

                x1_target = target.get("x1", 0.0)
                x2_target = target.get("x2", 0.0)
                if x1_target <= center_x_frame <= x2_target:
                    print(f"Bounding box touches center line ({x1_target:.1f} <= {center_x_frame} <= {x2_target:.1f}). Pausing alignment for {ALIGN_TOUCH_WAIT_S}s...")
                    stop(set_delay=False)
                    time.sleep(ALIGN_TOUCH_WAIT_S)
                    trigger_grab_sequence(cls)
                    continue

                centering_turns += 1
                print(f"In range ({dist:.1f}cm) but unaligned. Centering target (turn {centering_turns}/3): error={cx - center_x_frame:.1f}px")
                if cx > center_x_frame:
                    GPIO.output(IN1, GPIO.LOW)
                    GPIO.output(IN2, GPIO.HIGH)
                    GPIO.output(IN3, GPIO.LOW)
                    GPIO.output(IN4, GPIO.HIGH)
                else:
                    GPIO.output(IN1, GPIO.HIGH)
                    GPIO.output(IN2, GPIO.LOW)
                    GPIO.output(IN3, GPIO.HIGH)
                    GPIO.output(IN4, GPIO.LOW)
                set_speed(SEARCH_SPEED, bypass_delay=True)
                time.sleep(MOTOR_PULSE_S)
                stop(set_delay=False)
                time.sleep(MOTOR_WAIT_S)

                if centering_turns == 3:
                    print("Still not aligned after 2 turns. Centering completed. Proceeding with grab sequence.")
                    trigger_grab_sequence(cls)
                continue

def on_closing():
    stop()
    if stream is not None:
        try:
            stream.release()
        except:
            pass
    if hardware_available:
        try:
            pca.deinit()
        except:
            pass
    GPIO.cleanup()
    global_root.destroy()

def create_gui():
    global global_root, video_label
    root = tk.Tk()
    global_root = root
    root.title("Robot Control (Wheels & Servos)")
    root.geometry("950x420")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    motor_frame = tk.LabelFrame(root, text="Motor Control", padx=10, pady=10)
    motor_frame.pack(side="left", padx=15, pady=15, fill="y")

    btn_forward = tk.Button(motor_frame, text="Forward", command=forward_timed, width=10, height=2)
    btn_backward = tk.Button(motor_frame, text="Backward", command=backward_timed, width=10, height=2)
    btn_left = tk.Button(motor_frame, text="Left", command=left_timed, width=10, height=2)
    btn_right = tk.Button(motor_frame, text="Right", command=right_timed, width=10, height=2)
    btn_stop = tk.Button(motor_frame, text="Stop", command=stop_button_pressed, width=10, height=2, bg="red", fg="white")

    btn_forward.grid(row=0, column=1, padx=5, pady=5)
    btn_left.grid(row=1, column=0, padx=5, pady=5)
    btn_stop.grid(row=1, column=1, padx=5, pady=5)
    btn_right.grid(row=1, column=2, padx=5, pady=5)
    btn_backward.grid(row=2, column=1, padx=5, pady=5)

    video_frame = tk.LabelFrame(root, text="Camera Feed", padx=10, pady=10)
    video_frame.pack(side="left", padx=15, pady=15, fill="both", expand=True)

    video_label = tk.Label(video_frame, bg="black", width=320, height=240)
    video_label.pack(fill="both", expand=True)

    servo_frame = tk.LabelFrame(root, text="Servo Control", padx=10, pady=10)
    servo_frame.pack(side="right", padx=15, pady=15, fill="y")

    btn_start = tk.Button(servo_frame, text="Start (60, 0, 130, 90)", font=("Arial", 10), command=start)
    btn_start.pack(pady=10, fill="x", padx=10)

    btn_grab_pos = tk.Button(servo_frame, text="Grab Pos (All 90)", font=("Arial", 10), command=grab_pos)
    btn_grab_pos.pack(pady=10, fill="x", padx=10)

    btn_alt_grab_pos = tk.Button(servo_frame, text="Alt Grab Pos (75, 90, 90, 90)", font=("Arial", 10), command=alt_grab_pos)
    btn_alt_grab_pos.pack(pady=10, fill="x", padx=10)

    btn_grab = tk.Button(servo_frame, text="Grab (Last 180)", font=("Arial", 10), command=grab)
    btn_grab.pack(pady=10, fill="x", padx=10)

    global btn_auto
    btn_auto = tk.Button(servo_frame, text="Auto Mode: OFF", font=("Arial", 10), command=lambda: toggle_auto(btn_auto))
    btn_auto.pack(pady=10, fill="x", padx=10)

    def update_gui_image():
        try:
            frame = frame_queue.get_nowait()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            video_label.img_tk = img_tk
            video_label.config(image=img_tk)
        except queue.Empty:
            pass
        root.after(30, update_gui_image)

    stop()
    root.after(20, update_servo_task, root)
    root.after(20, update_gui_image)
    root.mainloop()

if __name__ == "__main__":
    if hardware_available and kit is not None:
        try:
            kit.servo[0].angle = 60
            kit.servo[1].angle = 0
            kit.servo[2].angle = 130
            kit.servo[3].angle = 90
        except Exception:
            pass

    current_angles[0] = 60.0
    current_angles[1] = 0.0
    current_angles[2] = 130.0
    current_angles[3] = 90.0

    auto_thread = threading.Thread(target=auto_loop, daemon=True)
    auto_thread.start()

    create_gui()
