# 🎯 Gaze Tracking and Analysis Toolkit

This toolkit enables real-time eye tracking using a regular webcam. It generates visualizations like gaze trails, heatmaps, AOI reports, and a full PDF summary — all through an interactive Streamlit dashboard.

---

## 📦 Features

- Upload an image or video stimulus
- Real-time gaze tracking with no additional hardware
- Automatic Area-of-Interest (AOI) duration and timeline analysis
- Gaze heatmap and replay video
- Exportable PDF report and CSV summaries
- User-friendly dashboard interface

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the dashboard**
   ```bash
   streamlit run gaze_tracker_dashboard.py
   ```

3. **Upload your stimulus**
   - Supported formats: `.jpg`, `.mp4`

4. **Run the pipeline**
   - Outputs:
     - `gaze_video.mp4`
     - `gaze_heatmap_with_aois.png`
     - `aoi_report.csv`
     - `aoi_timeline.csv`
     - `gaze_tracking_report.pdf`

---

## 📁 File Structure

- `run_pipeline.py`: Executes the complete pipeline
- `track_gaze_on_stimulus.py`: Captures gaze data
- `analyze_aoi.py`: Computes AOI durations and logs
- `heatmap_with_aois.py`: Creates gaze heatmap overlay
- `gaze_replay_video.py`: Generates replay video with gaze trail
- `gaze_tracker_dashboard.py`: Streamlit dashboard interface
- `requirements.txt`: Python dependencies

---

## 📌 Built by
**FACTSH Lab, IIIT Kottayam**  
For applications in digital humanities, film studies, and responsible computing.