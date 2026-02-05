import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import easyocr
import re

# ---------------- CONFIG ----------------
VIDEO_W, VIDEO_H = 900, 480

# ---------------- MODELS ----------------
vehicle_model = YOLO("yolov8n.pt")  # Vehicle detection
plate_model = YOLO("best.pt")       # Number plate detection
ocr = easyocr.Reader(['en'], gpu=False)

# ---------------- TKINTER UI ----------------
root = tk.Tk()
root.title("Vehicle & Number Plate Detection")
root.geometry("1200x650")
root.configure(bg="#2b2b2b")
root.resizable(False, False)

cap = None
running = False
current_image = None

# ---------------- UI ELEMENTS ----------------
tk.Label(root, text="Vehicle & Number Plate Detection", bg="#2b2b2b",
         fg="white", font=("Segoe UI", 14, "bold")).place(x=20, y=10)

video_label = tk.Label(root, bg="#444444")
video_label.place(x=260, y=80, width=VIDEO_W, height=VIDEO_H)

status = tk.Label(root, text="Ready", bg="#2b2b2b", fg="#aaaaaa")
status.place(x=10, y=620)

# ---------------- OCR CLEANING ----------------
def fix_plate_text(text):
    replacements = {'O':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8'}
    return ''.join(replacements.get(c.upper(), c.upper()) for c in text)

# ---------------- VIDEO/IMAGE ----------------
def choose_video():
    global cap, running, current_image
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if path:
        cap = cv2.VideoCapture(path)
        current_image = None
        running = True
        status.config(text="Video selected")
        update_video_frame()

def choose_image():
    global cap, running, current_image
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if path:
        img = cv2.imread(path)
        if img is None:
            status.config(text="Failed to open image!")
            return
        current_image = img
        cap = None
        running = True
        status.config(text="Image selected")
        process_image(current_image)

def stop_video():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    video_label.config(image="")
    status.config(text="Stopped")

# ---------------- PROCESS IMAGE ----------------
def process_image(frame):
    # Resize image to fit window
    h, w = frame.shape[:2]
    scale = min(VIDEO_W/w, VIDEO_H/h, 1)
    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

    # ---------- VEHICLE DETECTION (BLUE) ----------
    vehicle_results = vehicle_model(frame, conf=0.4, verbose=False)
    for r in vehicle_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in [2, 3, 5, 7]:  
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # ---------- PLATE DETECTION (GREEN) ----------
    plate_results = plate_model(frame, conf=0.25, verbose=False)
    for r in plate_results:
        for box in r.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            plate_img = frame[py1:py2, px1:px2]
            if plate_img.size == 0:
                continue

            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

            # OCR
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            ocr_results = ocr.readtext(gray)
            if not ocr_results:
                continue

            best_result = max(ocr_results, key=lambda x: x[2])
            raw_text = best_result[1]
            conf = best_result[2]
            if conf < 0.4:
                continue

            text = fix_plate_text(raw_text)

            # Draw text above box
            text_y = max(py1 - 10, 20)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (px1, text_y-th-8), (px1+tw+6, text_y), (0,255,0), -1)
            cv2.putText(frame, text, (px1+3, text_y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    display_frame(frame)

# ---------------- VIDEO LOOP ----------------
def update_video_frame():
    global cap, running
    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        stop_video()
        return

    frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    process_image(frame)  # reuse image processing

    if running:
        root.after(30, update_video_frame)

# ---------------- DISPLAY ----------------
def display_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

# ---------------- BUTTONS ----------------
tk.Button(root, text="Choose Video File", bg="#1e90ff", fg="white", width=20, command=choose_video).place(x=500, y=470)
tk.Button(root, text="Choose Image File", bg="#1e90ff", fg="white", width=20, command=choose_image).place(x=500, y=510)
tk.Button(root, text="Stop", bg="#d9534f", fg="white", width=15, command=stop_video).place(x=850, y=580)

root.mainloop()
