import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pynput import keyboard, mouse
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage



class ExamState:
    """Tracks the state of exam monitoring session"""
    def __init__(self):
        self.eye_movement_count = 0
        self.last_eye_alert = 0
        self.object_detection_cooldown = 0
        self.frame_counter = 0
        self.calibration_threshold = None

# Initialize models with optimized settings
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.7,
    model_selection=1  # For closer-range faces
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load YOLO model with medium size for better accuracy/speed balance
yolo_model = YOLO("yolov8m.pt")

# Tracking events
suspicious_activities = []
last_logged_time = {}
previous_frame = None
exam_state = ExamState()

def log_event(event):
    """Logs only unique events with a cooldown period"""
    global last_logged_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Cooldown logic - don't log same event within 10 seconds
    if event in last_logged_time and (datetime.now() - last_logged_time[event]).total_seconds() < 10:
        return
    
    last_logged_time[event] = datetime.now()
    suspicious_activities.append({"Time": timestamp, "Event": event})
    print(f"[ALERT] {event} detected at {timestamp}")

def calibrate_eye_threshold(cap, duration=5):
    """Calibrates eye movement thresholds during initial period"""
    print("Calibrating eye movement thresholds...")
    start_time = time.time()
    eye_values = []
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Left eye landmarks
                left_eye_outer = face_landmarks.landmark[33]
                left_eye_inner = face_landmarks.landmark[133]
                
                # Calculate eye aspect ratio
                eye_ar = abs(left_eye_outer.y - left_eye_inner.y) / abs(left_eye_outer.x - left_eye_inner.x)
                eye_values.append(eye_ar)
    
    if eye_values:
        threshold = np.mean(eye_values) * 0.7  # Use 70% of normal as threshold
        print(f"Calibration complete. Eye threshold set to {threshold:.2f}")
        return threshold
    
    print("Calibration failed. Using default threshold")
    return 0.25  # Default if calibration fails

def detect_face(frame):
    """Detects faces with improved confidence threshold"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    return results.detections is not None

def detect_eye_movement(frame):
    """Improved eye movement detection with dynamic thresholding"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get multiple eye landmarks for better accuracy
            left_eye_outer = face_landmarks.landmark[33]
            left_eye_inner = face_landmarks.landmark[133]
            right_eye_outer = face_landmarks.landmark[263]
            right_eye_inner = face_landmarks.landmark[362]
            
            # Calculate eye aspect ratios
            left_eye_ar = abs(left_eye_outer.y - left_eye_inner.y) / abs(left_eye_outer.x - left_eye_inner.x)
            right_eye_ar = abs(right_eye_outer.y - right_eye_inner.y) / abs(right_eye_outer.x - right_eye_inner.x)
            
            # Use calibrated threshold or dynamic fallback
            threshold = exam_state.calibration_threshold or 0.25
            
            if left_eye_ar < threshold or right_eye_ar < threshold:
                exam_state.eye_movement_count += 1
                
                # Only alert after multiple detections to reduce false positives
                if exam_state.eye_movement_count > 3 and (time.time() - exam_state.last_eye_alert) > 15:
                    log_event("Suspicious eye movement detected")
                    exam_state.last_eye_alert = time.time()
                    exam_state.eye_movement_count = 0
                    return True
    else:
        exam_state.eye_movement_count = max(0, exam_state.eye_movement_count - 1)
    
    return False

def detect_objects(frame):
    """Object detection with cooldown and confidence threshold"""
    if exam_state.object_detection_cooldown > 0:
        exam_state.object_detection_cooldown -= 1
        return False
        
    results = yolo_model(frame, conf=0.7, verbose=False)  # Increased confidence threshold
    
    suspicious_objects = []
    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls in ["cell phone", "book", "laptop"] and conf > 0.7:
                suspicious_objects.append(cls)
    
    if suspicious_objects:
        exam_state.object_detection_cooldown = 15  # 15-frame cooldown
        log_event(f"Foreign object detected: {', '.join(suspicious_objects)}")
        return True
    return False

def detect_movement(frame):
    """Improved movement detection with optical flow"""
    global previous_frame
    
    if previous_frame is None:
        previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return False

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow for better movement detection
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Calculate magnitude of movement
    magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    movement = np.mean(magnitude)
    
    previous_frame = current_gray
    
    # Only significant movements trigger further checks
    return movement > 2.0

class ActivityMonitor:
    """Monitors keyboard and mouse activity"""
    def __init__(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def on_key_press(self, key):
        try:
            # Only log character keys and ignore special keys
            if hasattr(key, 'char') and key.char and key.char.lower() != 'q':
                log_event("Keyboard activity detected")
        except AttributeError:
            pass

    def on_mouse_click(self, x, y, button, pressed):
        if pressed:
            log_event("Mouse activity detected")

def monitor_exam():
    """Main exam monitoring loop"""
    global previous_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Calibrate eye detection thresholds
    exam_state.calibration_threshold = calibrate_eye_threshold(cap)
    
    activity_monitor = ActivityMonitor()
    print("AI-Powered Exam Monitoring System Running...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            exam_state.frame_counter += 1
            
            # Skip some frames for performance (process every 2nd frame)
            if exam_state.frame_counter % 2 != 0:
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            movement_detected = detect_movement(frame)

            if movement_detected:
                if detect_face(frame):
                    # Only check eye movement periodically
                    if exam_state.frame_counter % 6 == 0:
                        detect_eye_movement(frame)
                
                # Only run object detection periodically
                if exam_state.frame_counter % 12 == 0:
                    detect_objects(frame)

            # Display status information
            status_text = [
                f"Movement: {'YES' if movement_detected else 'NO'}",
                f"Eye Alerts: {exam_state.eye_movement_count}",
                f"Obj Cooldown: {exam_state.object_detection_cooldown}"
            ]
            
            for i, text in enumerate(status_text):
                cv2.putText(frame, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("AI Exam Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        activity_monitor.keyboard_listener.stop()
        activity_monitor.mouse_listener.stop()

def generate_report():
    """Generates and saves a report of suspicious activities"""
    if not suspicious_activities:
        print("No suspicious activities to report")
        return

    df = pd.DataFrame(suspicious_activities)
    report_filename = f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_filename, index=False)
    print(f"[INFO] Report saved as {report_filename}")
    send_email(report_filename)

def send_email(report_filename):
    """Sends the report via email"""
    sender_email = "sarthak.sharma_cs22@gla.ac.in"
    receiver_email = "sarthak.sharma_cs22@gla.ac.in"
    subject = "Exam Monitoring Report"
    body = "Attached is the AI-powered exam monitoring report."

    try:
        msg = EmailMessage()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.set_content(body)

        with open(report_filename, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="csv",
                filename=report_filename
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, "your_app_password_here")
            server.send_message(msg)
        
        print("[INFO] Report emailed successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {str(e)}")

if __name__ == "__main__":
    print("Starting AI-Powered Exam Monitoring System...")
    monitor_exam()
    
    choice = input("Do you want the exam report? (yes/no): ").strip().lower()
    if choice == "yes":
        generate_report()
        print("Exam session completed. Report generated and emailed.")
    else:
        print("Exam session completed. No report generated.")