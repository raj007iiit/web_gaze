import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from collections import deque

st.set_page_config(page_title="WebRTC Gaze Tracker", layout="wide")
st.sidebar.title("🔧 Gaze Tracker Setup")
st.sidebar.markdown("If you are using this for the first time, please calibrate.")
calibrate = st.sidebar.button("🎯 Calibrate")

st.title("🎯 Real-Time Gaze Tracker with WebRTC")
st.markdown(
    '''
This dashboard uses `streamlit-webrtc` to track gaze using a webcam stream from your browser.
It uses MediaPipe face mesh landmarks and a trained regression model to estimate gaze positions on a 1280x720 screen.
Click 'Save Gaze Data' after running the tracker to store your gaze path as CSV.
'''
)

# Load gaze model
model = None
try:
    with open("gaze_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Calibration model not found. Please run gaze calibration first.")

SCREEN_W, SCREEN_H = 1280, 720

class GazeTracker(VideoTransformerBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.buffer = deque(maxlen=5)
        self.gaze_data = []

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks and model:
            face = results.multi_face_landmarks[0]
            try:
                eye_l = face.landmark[33]
                eye_r = face.landmark[133]
                eye_width = eye_r.x - eye_l.x
                eye_height = eye_r.y - eye_l.y

                iris_pts = [face.landmark[i] for i in range(468, 478)]
                iris_x = np.mean([p.x for p in iris_pts])
                iris_y = np.mean([p.y for p in iris_pts])

                rel_x = (iris_x - eye_l.x) / eye_width
                rel_y = (iris_y - eye_l.y) / eye_height

                pred_x, pred_y = model.predict([[rel_x, rel_y]])[0]
                pred_x = int(np.clip(pred_x, 0, SCREEN_W - 1))
                pred_y = int(np.clip(pred_y, 0, SCREEN_H - 1))

                self.buffer.append((pred_x, pred_y))
                smooth_x = int(np.mean([g[0] for g in self.buffer]))
                smooth_y = int(np.mean([g[1] for g in self.buffer]))

                cv2.circle(image, (smooth_x, smooth_y), 10, (0, 0, 255), -1)
                self.gaze_data.append((smooth_x, smooth_y))
            except:
                pass

        return image

ctx = webrtc_streamer(
    key="gaze-tracker",
   video_processor_factory=GazeTracker,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

if ctx and ctx.video_transformer:
    if st.button("💾 Save Gaze Data"):
        data = ctx.video_transformer.gaze_data
        if data:
            df = pd.DataFrame(data, columns=["x", "y"])
            df.to_csv("gaze_data.csv", index=False)
            st.success("✅ Gaze data saved to gaze_data.csv")
        else:
            st.warning("No gaze data captured yet.")
