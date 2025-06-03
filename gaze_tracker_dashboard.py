
import streamlit as st
import subprocess
import os
import time
from pathlib import Path
import zipfile

st.set_page_config(layout="wide")
st.title("🎯 Eye Tracker Gaze Dashboard")

# Introductory description
st.markdown("""
Welcome to the **Eye Tracker Gaze Dashboard** 👁️📊

This tool allows you to upload an image or video stimulus and visualize **where a user looked** over time using gaze heatmaps, AOI analysis, and replay videos. It uses real-time webcam-based eye landmark tracking powered by MediaPipe, combined with a trained model to estimate gaze positions.

> **Note:** If you're using this dashboard for the **first time**, make sure to run the **Gaze Calibration** process before using the tracker.
""")

# Upload image or video
uploaded_file = st.file_uploader("Upload a stimulus image or video", type=["jpg", "png", "mp4"])

if uploaded_file:
    # Remove any previously saved stimulus file
    for ext in [".jpg", ".jpeg", ".png", ".mp4", ".avi"]:
        old_path = f"stimulus_input{ext}"
        if os.path.exists(old_path):
            os.remove(old_path)

    # Save new uploaded file
    ext = Path(uploaded_file.name).suffix.lower()
    stimulus_path = f"stimulus_input{ext}"
    with open(stimulus_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded `{uploaded_file.name}` as `{stimulus_path}`")

    # Run pipeline
    if st.button("▶️ Run Gaze Tracking"):
        with st.spinner("Running pipeline..."):
            result = subprocess.run(["python", "run_pipeline.py"], capture_output=True, text=True)

        if result.returncode != 0:
            st.error("Pipeline failed!")
            st.text(result.stderr)
        else:
            st.success("✅ Pipeline completed successfully!")

            # Display video
            if os.path.exists("gaze_video.mp4"):
                st.video("gaze_video.mp4")

            # Display heatmap
            if os.path.exists("gaze_heatmap_with_aois.png"):
                st.image("gaze_heatmap_with_aois.png", caption="Gaze Heatmap")

            # Display AOI durations
            if os.path.exists("aoi_report.csv"):
                import pandas as pd
                df = pd.read_csv("aoi_report.csv")
                st.subheader("📊 AOI Duration")
                st.dataframe(df)

            # Report download
            if os.path.exists("gaze_tracking_report.pdf"):
                with open("gaze_tracking_report.pdf", "rb") as f:
                    st.download_button("📥 Download PDF Report", f, file_name="gaze_tracking_report.pdf")

            # Zip download
            output_files = [
                "gaze_data.csv",
                "gaze_heatmap_with_aois.png",
                "gaze_video.mp4",
                "aoi_report.csv",
                "aoi_timeline.csv",
                "gaze_tracking_report.pdf"
            ]
            if all(os.path.exists(f) for f in output_files):
                zip_name = "gaze_tracking_outputs.zip"
                with zipfile.ZipFile(zip_name, "w") as zipf:
                    for f in output_files:
                        zipf.write(f)
                with open(zip_name, "rb") as f:
                    st.download_button("📦 Download All Outputs (ZIP)", f, file_name=zip_name)
            else:
                st.warning("Some output files are missing. ZIP cannot be created.")

# Gaze Calibration Section
st.markdown("---")
if not os.path.exists("gaze_model.pkl"):
    st.warning("Gaze model not found. Please calibrate your gaze tracking before use.")
    if st.button("📍 Start Gaze Calibration"):
        with st.spinner("Running gaze calibration... Please follow the on-screen dots."):
            result = subprocess.run(["python", "gaze_calibration.py"], capture_output=True, text=True)

        if result.returncode != 0:
            st.error("Calibration failed!")
            st.text(result.stderr)
        else:
            st.success("✅ Calibration completed! You can now use the tracker.")


# Sidebar content
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. **Upload** a stimulus image (`.jpg`, `.png`) or video (`.mp4`)
    2. Click **▶️ Run Gaze Tracking**
    3. See the generated **heatmap**, **gaze video**, and **AOI metrics**
    4. Optionally **download** the report and outputs

    ---
    If you're using this for the **first time**, please calibrate your gaze by clicking below 👇
    """)

    if st.button("📍 Calibrate My Gaze (First Time Use)"):
        with st.spinner("Running gaze calibration... Please follow the on-screen dots."):
            result = subprocess.run(["python", "gaze_calibration.py"], capture_output=True, text=True)
        if result.returncode != 0:
            st.error("Calibration failed!")
            st.text(result.stderr)
        else:
            st.success("✅ Calibration completed! You can now proceed with tracking.")
