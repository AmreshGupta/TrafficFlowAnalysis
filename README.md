# Traffic Flow Analysis

## Objective

Analyze traffic flow by counting vehicles in three lanes using computer vision.

## Features

- Downloads and processes a YouTube traffic video
- Detects vehicles using YOLOv8 (COCO)
- Tracks vehicles with SORT
- Counts vehicles per lane (3 lanes)
- Exports results to CSV
- Overlays lane boundaries and real-time counts on video

## Setup

1. **Clone the repository**
2. **Install dependencies:**

   ```
   pip install opencv-python numpy pandas torch ultralytics yt-dlp
   ```

   - Download and add [SORT tracker](https://github.com/abewley/sort) to your project directory.

3. **Run the script:**

   ```
   python traffic_flow_analysis.py
   ```

4. **Output:**
   - `vehicle_counts.csv` with VehicleID, Lane, Frame, Timestamp
   - Video window with overlays (press `q` to quit)
   - Summary printed in terminal

## Demo Video

Record your screen while running the script and save as `demo.mp4`.

## Notes

- Adjust lane coordinates in the script as needed for your video.
- For best results, use a GPU.
