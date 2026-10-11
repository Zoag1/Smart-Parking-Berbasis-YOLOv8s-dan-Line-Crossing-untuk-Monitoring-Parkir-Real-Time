import cv2
import time
import numpy as np
import pandas as pd
import csv
import os
import argparse
from datetime import datetime
from ultralytics import YOLO

# ==================================================
# FORMAT TIME
# ==================================================
def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

# ==================================================
# BRIGHTNESS DETECTION
# ==================================================
def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# ==================================================
# LOW LIGHT ENHANCEMENT
# ==================================================
def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    gamma = 1.3
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(enhanced, table)

# ==================================================
# LINE INTERSECTION MATH
# ==================================================
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# ==================================================
# ARGUMENT
# ==================================================
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--model", default="yolov8s.pt")
args = vars(ap.parse_args())

# ==================================================
# CONFIG
# ==================================================
VIDEO_PATH = args["input"]
OUTPUT_VIDEO = args["output"]
MODEL_PATH = args["model"]
VIDEO_NAME = os.path.basename(VIDEO_PATH)

# ==================================================
# GROUND TRUTH LOADER
# ==================================================
GROUND_TRUTH_FILE = "ground_truth.csv"

actual_motor = 0
ground_truth_found = False

if os.path.exists(GROUND_TRUTH_FILE):
    try:
        gt_df = pd.read_csv(GROUND_TRUTH_FILE)

        match = gt_df.loc[
            gt_df["Video"] == VIDEO_NAME,
            "Actual_Motor"
        ]

        if not match.empty:
            actual_motor = int(match.iloc[0])
            ground_truth_found = True

    except Exception as e:
        print(f"[WARNING] Gagal membaca ground truth: {e}")
        
OUTPUT_CSV = "log_parkir_uksw.csv"
EVALUATION_CSV = "hasil_evaluasi.csv"

MAX_CAPACITY = 15
LOG_INTERVAL = 30
LIGHTING_MODE = "ADAPTIVE (CLAHE + Gamma 1.3 if Dark)" 
BRIGHTNESS_THRESHOLD = 90

# ==================================================
# PREPARATION
# ==================================================
output_dir = os.path.dirname(OUTPUT_VIDEO)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\n[INFO] Loading model...")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Video gagal dibuka")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_duration = format_seconds(total_frames / fps_input)
file_size_mb = (os.path.getsize(VIDEO_PATH) / (1024 * 1024))
process_start_clock = datetime.now()

LINE_START = (650, 850)
LINE_END = (1500, 850)

track_history = {}  
counted_ids = set() 
total_motor_masuk = 0
enhancement_used_count = 0
brightness_values = []
total_detected_boxes = 0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (frame_width, frame_height))

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Video", "Motor", "SisaSlot", "Status", "Inference_ms"])

# ==================================================
# INFO MULAI
# ==================================================
print("\n" + "=" * 65)
print("[INFO] MEMULAI PEMROSESAN")
print(f"Nama Video            : {VIDEO_NAME}")
print(f"Durasi Video          : {video_duration}")
print(f"Ukuran File           : {file_size_mb:.2f} MB")
print(f"Resolusi              : {frame_width}x{frame_height}")
print(f"FPS Video Asli        : {fps_input:.2f}")
print(f"Total Frame           : {total_frames}")
print(f"Waktu Mulai           : {process_start_clock.strftime('%H:%M:%S')}")
print(f"Counting Method       : LINE CROSSING (Garis)")
print(f"Peningkatan Cahaya    : {LIGHTING_MODE}")
print("=" * 65 + "\n")

cv2.namedWindow("Smart Parking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Parking", 480, 270)

frame_count = 0
start_global = time.time()
inference_times = []

# ==================================================
# MAIN LOOP
# ==================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame_count += 1
    current_video_time = format_seconds(frame_count / fps_input)
    
    # Adaptive Lighting Logic
    current_brightness = get_brightness(frame)
    brightness_values.append(current_brightness)
    if current_brightness < BRIGHTNESS_THRESHOLD:
        frame_proc = enhance_low_light(frame)
        active_mode = "ENHANCED"
        enhancement_used_count += 1
    else:
        frame_proc = frame
        active_mode = "ORIGINAL"

    t1 = time.time()
    
    # ByteTrack + IOU 0.5 (Solusi Double Box & Occlusion)
    results = model.track(
        frame_proc, 
        persist=True, 
        classes=[3], 
        conf=0.4,       
        iou=0.5,        
        tracker="bytetrack.yaml", 
        verbose=False
    )
    
    t2 = time.time()
    inference_ms = (t2 - t1) * 1000
    inference_times.append(inference_ms)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        total_detected_boxes += len(boxes)
        ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            if obj_id in track_history:
                prev_cx, prev_cy = track_history[obj_id]

                if obj_id not in counted_ids:
                    if intersect(LINE_START, LINE_END, (prev_cx, prev_cy), (cx, cy)):
                        if cy > prev_cy: 
                            total_motor_masuk += 1
                            counted_ids.add(obj_id)

            track_history[obj_id] = (cx, cy)

    slots_left = max(0, MAX_CAPACITY - total_motor_masuk)
    status = "PENUH" if slots_left <= 0 else "TERSEDIA"

    if frame_count % LOG_INTERVAL == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(OUTPUT_CSV, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, VIDEO_NAME, total_motor_masuk, slots_left, status, f"{inference_ms:.2f}"])

    # ==================================================
    # FULL PROGRESS INFO (GABUNGAN SEMUA)
    # ==================================================
    progress = (frame_count / total_frames) * 100
    elapsed = (time.time() - start_global)
    eta_val = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
    
    # Mencetak semua informasi dalam 1 baris print
    print(
        f"[{VIDEO_NAME}] "
        f"Progress: {progress:.1f}% | "
        f"Mode: {active_mode} | "
        f"Time: {current_video_time}/{video_duration} | "
        f"Motor: {total_motor_masuk} | "
        f"ETA: {eta_val:.1f}s ", 
        end="\r"
    )

    # Visualisasi
    annotated = results[0].plot()
    cv2.line(annotated, LINE_START, LINE_END, (0, 255, 255), 4)
    
    # UI Information Panel pada Frame Video
    cv2.rectangle(annotated, (10, 10), (480, 100), (0, 0, 0), -1)
    cv2.putText(annotated, f"Motor : {total_motor_masuk}/{MAX_CAPACITY}", (20, 35), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Sisa Slot : {slots_left}", (20, 65), 0, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Light Mode: {active_mode}", (20, 90), 0, 0.5, (255, 255, 0), 1)
    cv2.putText(annotated, f"Status : {status}", (250, 35), 0, 0.7, (0, 255, 0) if status == "TERSEDIA" else (0, 0, 255), 2)

    writer.write(annotated)
    cv2.imshow("Smart Parking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

# ==================================================
# RINGKASAN AKHIR
# ==================================================
cap.release()
writer.release()
cv2.destroyAllWindows()

process_end_clock = datetime.now()
process_total_time = (time.time() - start_global)
avg_inference = np.mean(inference_times)
avg_fps = (1000 / avg_inference) if avg_inference > 0 else 0
final_lighting_info = "Active (Video Detected as Dark)" if enhancement_used_count > (frame_count * 0.5) else "Inactive (Video Detected as Bright)"

avg_brightness = (
    np.mean(brightness_values)
    if len(brightness_values) > 0 else 0
)

count_error = abs(actual_motor - total_motor_masuk)

if actual_motor > 0:

    counting_accuracy = max(
        0,
        ((actual_motor - count_error) / actual_motor) * 100
    )

    error_rate = (
        count_error / actual_motor
    ) * 100

else:

    if total_motor_masuk == 0:
        counting_accuracy = 100
        error_rate = 0

    else:
        counting_accuracy = 0
        error_rate = 100

avg_detection_per_frame = (
    total_detected_boxes / frame_count
    if frame_count > 0 else 0
)

print("\n\n" + "=" * 65)
print("RINGKASAN AKHIR")
print(f"Nama Video            : {VIDEO_NAME}")
print(f"Durasi Video          : {video_duration}")
print(f"FPS Video Asli        : {fps_input:.2f}")
print(f"FPS Processing        : {avg_fps:.2f}")
print(f"Waktu Mulai           : {process_start_clock.strftime('%H:%M:%S')}")
print(f"Waktu Selesai         : {process_end_clock.strftime('%H:%M:%S')}")
print(f"Total Waktu Proses    : {format_seconds(process_total_time)}")
print(f"Peningkatan Cahaya    : {final_lighting_info}")
print(f"Motor Masuk Total     : {total_motor_masuk}")
print(f"Unique Track ID       : {len(counted_ids)}")

print("-" * 65)
print("HASIL EVALUASI")
print(f"Ground Truth Found    : {'YES' if ground_truth_found else 'NO'}")
print(f"Jumlah Aktual Motor   : {actual_motor}")
print(f"Jumlah Sistem         : {total_motor_masuk}")
print(f"Counting Error        : {count_error}")
print(f"Counting Accuracy     : {counting_accuracy:.2f}%")
print(f"Error Rate            : {error_rate:.2f}%")
print(f"Average Inference     : {avg_inference:.2f} ms")
print(f"Average Brightness    : {avg_brightness:.2f}")
print(f"Total Bounding Box    : {total_detected_boxes}")
print(f"Detection / Frame     : {avg_detection_per_frame:.2f}")
print(f"Frame Diproses        : {frame_count}")
print(f"Track History         : {len(track_history)}")

enhancement_percentage = (
    enhancement_used_count / frame_count * 100
) if frame_count > 0 else 0

print(f"CLAHE Usage           : {enhancement_percentage:.2f}%")
print(f"Output Video          : {OUTPUT_VIDEO}")
print(f"Output CSV            : {OUTPUT_CSV}")
if not os.path.exists(EVALUATION_CSV):
    with open(EVALUATION_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "Video",
            "Actual_Motor",
            "System_Count",
            "Counting_Error",
            "Counting_Accuracy",
            "Error_Rate",
            "FPS",
            "Inference_ms",
            "Brightness"
        ])

with open(EVALUATION_CSV, "a", newline="") as f:
    csv.writer(f).writerow([
        VIDEO_NAME,
        actual_motor,
        total_motor_masuk,
        count_error,
        f"{counting_accuracy:.2f}",
        f"{error_rate:.2f}",
        f"{avg_fps:.2f}",
        f"{avg_inference:.2f}",
        f"{avg_brightness:.2f}"
    ])
print("=" * 65)
print("[INFO] Tugas selesai.")