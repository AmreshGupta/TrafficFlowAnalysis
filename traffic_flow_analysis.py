import cv2
import numpy as np
import torch
import pandas as pd
import time
import os
from collections import defaultdict
from ultralytics import YOLO
from sort import Sort  # You need to download SORT: https://github.com/abewley/sort
import yt_dlp

# Download video from YouTube
def download_video(url, output_path):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Define lane boundaries (customize as per your video)
def get_lane(x, y, lanes):
    for idx, (x1, y1, x2, y2) in enumerate(lanes):
        if x1 < x < x2 and y1 < y < y2:
            return idx + 1
    return None

def main():
    # 1. Download video
    video_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
    video_path = "traffic.mp4"
    if not os.path.exists(video_path):
        print("Downloading video...")
        download_video(video_url, video_path)

    # 2. Load YOLOv5 model
    model = YOLO('yolov8n.pt')  # Or yolov5s.pt, yolov8n.pt, etc.

    # 3. Initialize SORT tracker
    tracker = Sort()

    # 4. Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    # 5. Define lanes (x1, y1, x2, y2) - adjust these for your video
    lanes = [
        (40, 300, 200, 700),   # Lane 1 (aur left)
        (210, 300, 370, 700),  # Lane 2 (aur left)
        (380, 300, 540, 700),  # Lane 3 (aur left)
    ]

    # 6. Vehicle classes in COCO
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # 7. Data structures
    vehicle_lane = dict()
    vehicle_frame = dict()
    vehicle_time = dict()
    lane_counts = defaultdict(set)
    csv_rows = []

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        timestamp = frame_count / fps

        # Detect vehicles
        results = model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

        # Track vehicles
        dets = np.array(detections)
        tracks = tracker.update(dets)

        # Draw lanes
        for idx, (x1, y1, x2, y2) in enumerate(lanes):
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Lane {idx+1}", (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Count and annotate
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            lane = get_lane(cx, cy, lanes)
            if lane:
                lane_counts[lane].add(track_id)
                vehicle_lane[track_id] = lane
                vehicle_frame[track_id] = frame_count
                vehicle_time[track_id] = timestamp
                csv_rows.append([track_id, lane, frame_count, timestamp])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID:{track_id} L:{lane}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Overlay counts
        for idx in range(3):
            cv2.putText(frame, f"Lane {idx+1} Count: {len(lane_counts[idx+1])}", (50, 50+40*idx), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)

        # Show frame (optional, for demo)
        cv2.imshow("Traffic Flow Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 8. Export CSV
    df = pd.DataFrame(csv_rows, columns=["VehicleID", "Lane", "Frame", "Timestamp"])
    df.to_csv("vehicle_counts.csv", index=False)

    # 9. Print summary
    print("\nSummary:")
    for idx in range(3):
        print(f"Lane {idx+1}: {len(lane_counts[idx+1])} vehicles")

if __name__ == "__main__":
    main()