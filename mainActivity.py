import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from threading import Thread
import os
from time import time
import numpy as np

from ultralytics import YOLO
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

import easyocr

# ===================== SPEED CALIBRATION =====================
REAL_DISTANCE_METERS = 10     # real distance on road
PIXEL_DISTANCE = 200          # pixel distance between same points
METERS_PER_PIXEL = REAL_DISTANCE_METERS / PIXEL_DISTANCE


# ========================== UI CLASS ==========================
class SpeedMonitoringUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speed & Number Plate Monitoring System")
        self.root.geometry("900x600")
        self.root.configure(bg="#2E2E2E")

        self.video_path = None
        self.detector = None
        self.running = False

        tk.Label(
            root,
            text="Speed & Number Plate Monitoring System",
            font=("Arial", 18, "bold"),
            bg="#2E2E2E",
            fg="white"
        ).pack(pady=10)

        main = tk.Frame(root, bg="#2E2E2E")
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg="#2E2E2E")
        left.pack(side="left", fill="both", expand=True)

        self.preview = tk.Label(
            left, text="No video selected",
            bg="#4A4A4A", fg="lightgray",
            width=60, height=20
        )
        self.preview.pack(pady=10)

        tk.Button(
            left, text="Select Video",
            bg="#007BFF", fg="white",
            command=self.select_video
        ).pack()

        self.file_label = tk.Label(left, bg="#2E2E2E", fg="lightgray")
        self.file_label.pack(pady=5)

        right = tk.Frame(main, bg="#3C3C3C", relief="groove", borderwidth=2)
        right.pack(side="right", fill="y", padx=10)

        tk.Label(right, text="Settings", font=("Arial", 14, "bold"),
                 bg="#3C3C3C", fg="white").pack(pady=10)

        tk.Label(right, text="Speed Limit (km/h)", bg="#3C3C3C", fg="white").pack()
        self.speed_limit = tk.Entry(right)
        self.speed_limit.insert(0, "40")
        self.speed_limit.pack()

        tk.Label(right, text="YOLO Model", bg="#3C3C3C", fg="white").pack()
        self.model_var = tk.StringVar(value="yolov8n.pt")
        ttk.Combobox(
            right,
            textvariable=self.model_var,
            values=["yolov8n.pt"],
            state="readonly"
        ).pack()

        footer = tk.Frame(root, bg="#2E2E2E")
        footer.pack(fill="x", pady=10)

        self.status = tk.Label(footer, text="Ready", bg="#2E2E2E", fg="lightgray")
        self.status.pack(side="left", padx=10)

        self.start_btn = tk.Button(
            footer, text="Start",
            bg="#28A745", fg="white",
            state="disabled",
            command=self.start
        )
        self.start_btn.pack(side="right", padx=10)

        self.stop_btn = tk.Button(
            footer, text="Stop",
            bg="#DC3545", fg="white",
            state="disabled",
            command=self.stop
        )
        self.stop_btn.pack(side="right")

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("MP4 files", "*.mp4")]
        )
        if self.video_path:
            self.file_label.config(text=os.path.basename(self.video_path))
            self.start_btn.config(state="normal")

    def start(self):
        try:
            limit = float(self.speed_limit.get())
        except:
            messagebox.showerror("Error", "Invalid speed limit")
            return

        self.detector = SpeedEstimator(
            model=self.model_var.get(),
            speed_limit=limit
        )

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        Thread(target=self.process_video, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_id = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % 3 != 0:
                continue

            frame = cv2.resize(frame, (640, 360))
            output = self.detector.estimate_speed(frame)

            cv2.imshow("Speed & Number Plate Detection", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.stop()


# ========================== DETECTION CLASS ==========================
class SpeedEstimator(BaseSolution):
    def __init__(self, speed_limit=40, **kwargs):
        super().__init__(**kwargs)

        self.speed_limit = speed_limit
        self.prev_pos = {}
        self.prev_time = {}
        self.speed_buffer = {}
        self.plate_cache = {}

        self.ocr = easyocr.Reader(['en'], gpu=False)

    def estimate_speed(self, frame):
        self.annotator = Annotator(frame)

        results = self.model.track(
            frame,
            persist=True,
            conf=0.4,
            classes=[2, 3, 5, 7]
        )

        if results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        now = time()

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # =================== CORRECTED SPEED LOGIC ===================
            speed = 0.0

            if tid in self.prev_pos:
                px, py = self.prev_pos[tid]
                pixel_dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                dt = now - self.prev_time[tid]

                if dt > 0 and pixel_dist > 2:
                    dist_m = pixel_dist * METERS_PER_PIXEL
                    instant_speed = (dist_m / dt) * 3.6

                    alpha = 0.3
                    prev_speed = self.speed_buffer.get(tid, instant_speed)
                    speed = alpha * instant_speed + (1 - alpha) * prev_speed

                    self.speed_buffer[tid] = speed
                else:
                    speed = self.speed_buffer.get(tid, 0.0)

            speed = int(speed)
            # ============================================================

            self.prev_pos[tid] = (cx, cy)
            self.prev_time[tid] = now

            plate = ""
            if tid not in self.plate_cache and speed > 5:
                crop = frame[y2 - 40:y2, x1:x2]
                if crop.size > 0:
                    txt = self.ocr.readtext(
                        crop, detail=0,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )
                    plate = txt[0] if txt else ""
                self.plate_cache[tid] = plate
            else:
                plate = self.plate_cache.get(tid, "")

            label = f"ID:{tid} | {speed} km/h | {plate}"
            color = (0, 0, 255) if speed > self.speed_limit else colors(tid, True)
            self.annotator.box_label(box, label, color=color)

        return frame


# ========================== MAIN ==========================
if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedMonitoringUI(root)
    root.mainloop()